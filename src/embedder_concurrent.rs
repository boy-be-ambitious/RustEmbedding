use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use ort::{
    ep,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use parking_lot::RwLock;
use rayon::prelude::*;

use crate::tokenizer::EtsTokenizer;

const HIDDEN_DIM: usize = 768;

pub struct ConcurrentEmbedder {
    session: Arc<RwLock<Session>>,
    tokenizer: Arc<RwLock<EtsTokenizer>>,
    pub dim: usize,
}

unsafe impl Send for ConcurrentEmbedder {}
unsafe impl Sync for ConcurrentEmbedder {}

impl ConcurrentEmbedder {
    pub fn load(model_dir: &Path, ort_lib: &Path) -> Result<Self> {
        let _ort_lib_str = ort_lib
            .to_str()
            .context("ort_lib path is not valid UTF-8")?;
        log::info!("  [1/3] Loading ORT via LoadLibraryW (bypassing libloading) ...");

        #[cfg(windows)]
        {
            use windows::core::PCWSTR;
            use windows::Win32::System::LibraryLoader::{
                GetProcAddress, LoadLibraryW, SetDllDirectoryW,
            };

            let abs = if ort_lib.is_absolute() {
                ort_lib.to_path_buf()
            } else {
                std::env::current_dir()
                    .context("Cannot get current dir")?
                    .join(ort_lib)
            };
            let dll_dir = abs
                .parent()
                .ok_or_else(|| anyhow::anyhow!("Cannot determine parent dir of ort_lib"))?;

            let dir_wide: Vec<u16> = dll_dir
                .to_str()
                .context("dll dir path not valid UTF-8")?
                .encode_utf16()
                .chain(std::iter::once(0u16))
                .collect();
            unsafe { SetDllDirectoryW(PCWSTR(dir_wide.as_ptr())) }
                .context("SetDllDirectoryW failed")?;

            let abs_wide: Vec<u16> = abs
                .to_str()
                .context("ort_lib absolute path not valid UTF-8")?
                .encode_utf16()
                .chain(std::iter::once(0u16))
                .collect();
            let hmod = unsafe { LoadLibraryW(PCWSTR(abs_wide.as_ptr())) }
                .context("LoadLibraryW failed to load onnxruntime.dll")?;

            log::info!("  [1/3] dll loaded, resolving OrtGetApiBase ...");

            type OrtGetApiBaseFn = unsafe extern "C" fn() -> *const ort_sys::OrtApiBase;
            let get_base: OrtGetApiBaseFn = unsafe {
                let proc = GetProcAddress(hmod, windows::core::s!("OrtGetApiBase"))
                    .ok_or_else(|| anyhow::anyhow!("OrtGetApiBase not found in dll"))?;
                std::mem::transmute(proc)
            };

            let base = unsafe { get_base() };
            anyhow::ensure!(!base.is_null(), "OrtGetApiBase returned null");

            let get_api = unsafe { (*base).GetApi };
            let api_ptr = unsafe { get_api(ort_sys::ORT_API_VERSION) };
            anyhow::ensure!(
                !api_ptr.is_null(),
                "GetApi({}) returned null",
                ort_sys::ORT_API_VERSION
            );

            let api = unsafe { std::ptr::read(api_ptr) };
            ort::set_api(api);
            log::info!(
                "  [1/3] ORT API injected (version {}).",
                ort_sys::ORT_API_VERSION
            );
        }

        #[cfg(not(windows))]
        {
            ort::init_from(_ort_lib_str)
                .context("Failed to load onnxruntime")?
                .with_telemetry(false)
                .commit();
        }

        log::info!("  [2/3] Creating session builder ...");
        let builder = Session::builder().context("Failed to create ONNX session builder")?;
        log::info!("  [2/3] Setting CPU execution provider ...");
        let builder = builder
            .with_execution_providers([ep::CPU::default().build()])
            .map_err(|e| anyhow::anyhow!("Failed to set CPU execution provider: {}", e))?;
        log::info!("  [2/3] Disabling graph optimisation ...");
        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(|e| anyhow::anyhow!("Failed to set optimisation level: {}", e))?;
        let model_path = model_dir.join("model_fp16.onnx");
        log::info!("  [2/3] Calling commit_from_file on {:?} ...", model_path);
        let session = builder
            .commit_from_file(&model_path)
            .with_context(|| format!("Failed to load ONNX model from {:?}", model_path))?;
        log::info!("  [2/2] Model loaded.");

        log::info!("  [3/3] Loading tokenizer ...");
        let tokenizer = EtsTokenizer::load(model_dir)?;

        Ok(Self {
            session: Arc::new(RwLock::new(session)),
            tokenizer: Arc::new(RwLock::new(tokenizer)),
            dim: HIDDEN_DIM,
        })
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let (input_ids, attention_mask, _token_type_ids, seq_len) = {
            let tokenizer = self.tokenizer.read();
            tokenizer.encode_batch(texts)?
        };

        let batch = texts.len();

        let ids_tensor =
            Tensor::<i64>::from_array((vec![batch as i64, seq_len as i64], input_ids.clone()))
                .context("Failed to create input_ids tensor")?;

        let mask_tensor =
            Tensor::<i64>::from_array((vec![batch as i64, seq_len as i64], attention_mask.clone()))
                .context("Failed to create attention_mask tensor")?;

        let (raw_vec, hidden_size) = {
            let mut session = self.session.write();
            let outputs = session
                .run(ort::inputs![ids_tensor, mask_tensor])
                .context("ONNX inference failed")?;
            let lhs = outputs[0]
                .try_extract_tensor::<f32>()
                .context("Failed to extract last_hidden_state tensor")?;
            let raw = lhs.1;
            let hs = if raw.len() > 0 {
                raw.len() / (batch * seq_len)
            } else {
                HIDDEN_DIM
            };
            (raw.to_vec(), hs)
        };

        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(batch);

        for b in 0..batch {
            let mut vec = vec![0f32; hidden_size];
            let mut count = 0f32;
            for s in 0..seq_len {
                if attention_mask[b * seq_len + s] == 1 {
                    let base = (b * seq_len + s) * hidden_size;
                    for h in 0..hidden_size {
                        vec[h] += raw_vec[base + h];
                    }
                    count += 1.0;
                }
            }
            if count > 0.0 {
                for v in &mut vec {
                    *v /= count;
                }
            }
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-9 {
                for v in &mut vec {
                    *v /= norm;
                }
            }
            embeddings.push(vec);
        }

        Ok(embeddings)
    }

    pub fn embed_batch_parallel(
        &self,
        texts: &[&str],
        num_threads: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let chunk_size = (texts.len() + num_threads - 1) / num_threads;
        let chunks: Vec<&[&str]> = texts.chunks(chunk_size).collect();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .context("Failed to create thread pool")?;

        let results: Result<Vec<Vec<Vec<f32>>>> = pool.install(|| {
            chunks
                .into_par_iter()
                .map(|chunk| self.embed_batch(chunk))
                .collect()
        });

        Ok(results?.into_iter().flatten().collect())
    }
}
