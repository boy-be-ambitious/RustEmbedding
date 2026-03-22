//! ONNX Runtime session wrapper for VESO-25M (FP16).
//!
//! Loads the model once, then exposes `embed_batch` which runs inference on a
//! slice of raw texts and returns L2-normalised float32 vectors.
//!
//! The ONNX graph expects three inputs:
//!   - `input_ids`      [batch, seq_len]  int64
//!   - `attention_mask` [batch, seq_len]  int64
//!   - `token_type_ids` [batch, seq_len]  int64
//!
//! The graph produces a `last_hidden_state` output of shape
//! [batch, seq_len, hidden_size].  We apply masked mean-pooling to obtain one
//! vector per sequence, then L2-normalise.

use std::path::Path;

use anyhow::{Context, Result};
use ort::{
    ep,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
#[cfg(windows)]
use ort_sys;

use crate::tokenizer::EtsTokenizer;

/// VESO-25M hidden dimension (fixed by the model architecture).
const HIDDEN_DIM: usize = 768;

pub struct Embedder {
    session: Session,
    tokenizer: EtsTokenizer,
    pub dim: usize,
}

impl Embedder {
    /// Load the ONNX model and tokenizer from `model_dir`.
    /// `ort_lib` is the path to `onnxruntime.dll` (Windows) / `libonnxruntime.so`.
    pub fn load(model_dir: &Path, ort_lib: &Path) -> Result<Self> {
        // Pre-load the DLL via Windows LoadLibraryW so that by the time
        // libloading (used by ort::init_from) calls LoadLibraryExW the handle
        // is already cached in the process loader and returns instantly.
        let ort_lib_str = ort_lib
            .to_str()
            .context("ort_lib path is not valid UTF-8")?;
        log::info!("  [1/3] Loading ORT via LoadLibraryW (bypassing libloading) ...");

        // On Windows, libloading uses LoadLibraryExW with LOAD_WITH_ALTERED_SEARCH_PATH,
        // which triggers Windows Smart App Control reputation checks and hangs.
        // We use SetDllDirectoryW + LoadLibraryW(filename only) to bypass that,
        // then inject the OrtApi pointer directly via ort::set_api.
        #[cfg(windows)]
        {
            use windows::core::PCWSTR;
            use windows::Win32::System::LibraryLoader::{
                GetProcAddress, LoadLibraryW, SetDllDirectoryW,
            };

            // Resolve absolute path (avoid \\?\ prefix which LoadLibraryW rejects).
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

            // Add the dll's directory to the search path so its own dependencies load.
            let dir_wide: Vec<u16> = dll_dir
                .to_str()
                .context("dll dir path not valid UTF-8")?
                .encode_utf16()
                .chain(std::iter::once(0u16))
                .collect();
            unsafe { SetDllDirectoryW(PCWSTR(dir_wide.as_ptr())) }
                .context("SetDllDirectoryW failed")?;

            // Load by full absolute path to guarantee the exact dll is used.
            let abs_wide: Vec<u16> = abs
                .to_str()
                .context("ort_lib absolute path not valid UTF-8")?
                .encode_utf16()
                .chain(std::iter::once(0u16))
                .collect();
            let hmod = unsafe { LoadLibraryW(PCWSTR(abs_wide.as_ptr())) }
                .context("LoadLibraryW failed to load onnxruntime.dll")?;

            log::info!("  [1/3] dll loaded, resolving OrtGetApiBase ...");

            // Resolve OrtGetApiBase
            type OrtGetApiBaseFn = unsafe extern "C" fn() -> *const ort_sys::OrtApiBase;
            let get_base: OrtGetApiBaseFn = unsafe {
                let proc = GetProcAddress(hmod, windows::core::s!("OrtGetApiBase"))
                    .ok_or_else(|| anyhow::anyhow!("OrtGetApiBase not found in dll"))?;
                std::mem::transmute(proc)
            };

            let base = unsafe { get_base() };
            anyhow::ensure!(!base.is_null(), "OrtGetApiBase returned null");

            // Get the OrtApi for the required API version.
            let get_api = unsafe { (*base).GetApi };
            let api_ptr = unsafe { get_api(ort_sys::ORT_API_VERSION) };
            anyhow::ensure!(
                !api_ptr.is_null(),
                "GetApi({}) returned null — dll version may be incompatible",
                ort_sys::ORT_API_VERSION
            );

            // Inject into ort's global API slot.
            let api = unsafe { std::ptr::read(api_ptr) };
            ort::set_api(api);
            log::info!(
                "  [1/3] ORT API injected (version {}).",
                ort_sys::ORT_API_VERSION
            );
        }

        #[cfg(not(windows))]
        {
            // On non-Windows platforms just use the normal init_from path.
            ort::init_from(ort_lib_str)
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
        log::info!("  [2/3] Model loaded.");

        log::info!("  [3/3] Loading tokenizer ...");
        let tokenizer = EtsTokenizer::load(model_dir)?;

        Ok(Self {
            session,
            tokenizer,
            dim: HIDDEN_DIM,
        })
    }

    /// Embed a batch of texts. Returns a `Vec` of length `texts.len()`, each
    /// element being a `Vec<f32>` of length `self.dim`.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let (input_ids, attention_mask, _token_type_ids, seq_len) =
            self.tokenizer.encode_batch(texts)?;

        let batch = texts.len();

        // Build ONNX tensors — VESO-25M takes only input_ids + attention_mask.
        let ids_tensor = Tensor::<i64>::from_array((vec![batch as i64, seq_len as i64], input_ids))
            .context("Failed to create input_ids tensor")?;

        let mask_tensor =
            Tensor::<i64>::from_array((vec![batch as i64, seq_len as i64], attention_mask.clone()))
                .context("Failed to create attention_mask tensor")?;

        // Run inference.
        let outputs = self
            .session
            .run(ort::inputs![ids_tensor, mask_tensor])
            .context("ONNX inference failed")?;

        // Extract last_hidden_state: shape [batch, seq_len, hidden_size].
        let lhs = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract last_hidden_state tensor")?;

        let raw = lhs.1;

        // Masked mean-pool over the sequence dimension.
        let hidden_size = if raw.len() > 0 {
            raw.len() / (batch * seq_len)
        } else {
            HIDDEN_DIM
        };

        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(batch);

        for b in 0..batch {
            let mut vec = vec![0f32; hidden_size];
            let mut count = 0f32;
            for s in 0..seq_len {
                if attention_mask[b * seq_len + s] == 1 {
                    let base = (b * seq_len + s) * hidden_size;
                    for h in 0..hidden_size {
                        vec[h] += raw[base + h];
                    }
                    count += 1.0;
                }
            }
            if count > 0.0 {
                for v in &mut vec {
                    *v /= count;
                }
            }
            // L2 normalise.
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
}
