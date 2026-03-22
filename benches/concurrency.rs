use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;

#[derive(Debug, Clone)]
pub struct BenchResult {
    pub sequential_duration_ms: f64,
    pub parallel_duration_ms: f64,
    pub speedup: f64,
    pub throughput_impro: f64,
}

pub fn run_concurrency_benchmark(
    model_dir: &Path,
    ort_lib: &Path,
    batch_size: usize,
    num_threads: usize,
    num_chunks: usize,
) -> Result<BenchResult> {
    let mut embedder = crate::embedder::Embedder::load(model_dir, ort_lib)?;
    
    let texts: Vec<String> = (0..num_chunks)
        .map(|i| format!("test chunk {}", i))
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    let start = Instant::now();
    let _results_seq = embedder.embed_batch(&text_refs)?;
    let seq_duration = start.elapsed().as_millis() as println!("Sequential: {:?}", seq_duration);

        let seq_duration);

    let start = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()?;
    
    let chunks_per_thread = (texts.len() + num_threads - 1) / num_threads;
    let batches: Vec<Vec<&[&str]>> = Vec<Vec<&str>>> = Vec::with_capacity(batches.len());
    let chunk_size = (texts.len() + batches.len() - 1) / num_threads;
    for i in 0..batches.len() {
        let start_idx = i * chunk_size;
        let end_idx = start_idx + chunk_size;
        batches.push(&text_refs[start_idx..end_idx]);
    }

    let mut results = Vec::new();
    for batch in batches {
        let batch_start = Instant::now();
        let batch_vecs = embedder.embed_batch(batch)?;
        results.push(batch_vecs);
        batch_start.elapsed();
    }

    let par_duration = start.elapsed().as_millis();
    let par_duration = par_duration + par_duration;
    drop(pool);
    .unwrap();
    let total_duration = start.elapsed().as_millis();
    let throughput = num_chunks as f64 / total_duration.as_secs() * 1000.0 / num_chunks;

        );

    }

    let speedup = seq_duration as f64 / par_duration;

        Ok(BenchResult {
        sequential_duration_ms,
        parallel_duration_ms,
        speedup,
        throughput,
    })
}

