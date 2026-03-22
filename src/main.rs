//! CLI entry point for RustEmbedding.
//!
//! Sub-commands:
//!   build  – chunk a repo, embed all chunks, write a binary index
//!   search – load an existing index and answer a natural-language query

mod chunker;
mod embedder;
mod index;
mod perf;
mod tokenizer;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use log::{info, warn};

use crate::embedder::Embedder;
use crate::index::{Entry, Index};
use crate::perf::{memory_snapshot, BuildReport, MemoryReport, Timer, TimingMs};

// ─────────────────────────── CLI definition ──────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "rust-embedding",
    about = "Semantic code search for ArkTS / HarmonyOS .ets files",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Build a vector index from a repo directory.
    Build(BuildArgs),
    /// Query an existing index.
    Search(SearchArgs),
}

#[derive(Parser, Debug)]
struct BuildArgs {
    /// Root directory to scan for .ets files.
    #[arg(long, default_value = "hmosworld-master")]
    repo: PathBuf,

    /// Directory containing tokenizer.json and model_fp16.onnx.
    #[arg(long)]
    model: PathBuf,

    /// Path to onnxruntime.dll (Windows) / libonnxruntime.so.
    #[arg(long)]
    ort: PathBuf,

    /// Output binary index file.
    #[arg(long, default_value = "index.bin")]
    out: PathBuf,

    /// ONNX inference batch size (16–32 recommended).
    #[arg(long, default_value_t = 16)]
    batch: usize,

    /// Stem for report files (writes <stem>.json and <stem>.md).
    /// Leave empty to skip report generation.
    #[arg(long, default_value = "")]
    report: String,
}

#[derive(Parser, Debug)]
struct SearchArgs {
    /// Natural-language or code query.
    #[arg(long)]
    query: String,

    /// Index file produced by `build`.
    #[arg(long, default_value = "index.bin")]
    index: PathBuf,

    /// Directory containing tokenizer.json and model_fp16.onnx.
    #[arg(long)]
    model: PathBuf,

    /// Path to onnxruntime.dll.
    #[arg(long)]
    ort: PathBuf,

    /// Number of results to return.
    #[arg(long, default_value_t = 5)]
    top: usize,
}

// ─────────────────────────── main ────────────────────────────────────────────

fn main() -> Result<()> {
    // Initialise logger (RUST_LOG=info by default).
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    let cli = Cli::parse();

    match cli.command {
        Command::Build(args) => cmd_build(args),
        Command::Search(args) => cmd_search(args),
    }
}

// ─────────────────────────── build ───────────────────────────────────────────

fn cmd_build(args: BuildArgs) -> Result<()> {
    let total_timer = Timer::start();
    let mem_before = memory_snapshot();

    // ── 1. Chunk ──────────────────────────────────────────────────────────────
    info!("Chunking .ets files under {:?} ...", args.repo);
    let chunk_timer = Timer::start();
    let chunks = chunker::chunk_repo(&args.repo)
        .with_context(|| format!("Failed to chunk repo {:?}", args.repo))?;
    let chunking_ms = chunk_timer.elapsed_ms();

    // Count distinct files.
    let mut files: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    for c in &chunks {
        files.insert(c.file.clone());
    }
    let total_files = files.len();
    let total_chunks = chunks.len();

    info!(
        "  {} files → {} chunks  ({:.2}ms)",
        total_files, total_chunks, chunking_ms as f64
    );

    if total_chunks == 0 {
        warn!("No chunks found. Is the repo path correct?");
        return Ok(());
    }

    // ── 2. Load model ─────────────────────────────────────────────────────────
    info!("Loading tokenizer and ONNX model ...");
    let model_timer = Timer::start();
    let mut embedder = Embedder::load(&args.model, &args.ort)?;
    let model_load_ms = model_timer.elapsed_ms();
    info!("  Model loaded ({:.2}s).", model_load_ms as f64 / 1000.0);

    // ── 3. Embed ──────────────────────────────────────────────────────────────
    info!(
        "Embedding {} chunks (batch_size={}) ...",
        total_chunks, args.batch
    );
    let embed_timer = Timer::start();

    let index_build_timer;
    let mut idx = Index::new(embedder.dim);

    {
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let mut done = 0usize;

        for batch_start in (0..total_chunks).step_by(args.batch) {
            let batch_end = (batch_start + args.batch).min(total_chunks);
            let batch_texts = &texts[batch_start..batch_end];
            let vecs = embedder.embed_batch(batch_texts)?;

            for (i, vec) in vecs.into_iter().enumerate() {
                let chunk = &chunks[batch_start + i];
                idx.add(Entry {
                    file: chunk.file.clone(),
                    start_line: chunk.start_line,
                    end_line: chunk.end_line,
                    text: chunk.text.clone(),
                    vector: vec,
                });
            }

            done += batch_end - batch_start;
            if done % 50 < args.batch {
                info!("  {}/{} chunks embedded", done, total_chunks);
            }
        }
    }

    let embedding_ms = embed_timer.elapsed_ms();
    info!("  Embedding done ({:.2}s)", embedding_ms as f64 / 1000.0);

    // ── 4. Build index (already done inline above; just time the save) ────────
    index_build_timer = Timer::start();
    let index_build_ms = index_build_timer.elapsed_ms(); // nearly zero

    let save_timer = Timer::start();
    idx.save(&args.out)
        .with_context(|| format!("Failed to save index to {:?}", args.out))?;
    let index_save_ms = save_timer.elapsed_ms();

    info!("Index saved to {:?}", args.out);

    // ── 5. Collect metrics ────────────────────────────────────────────────────
    let total_ms = total_timer.elapsed_ms();
    let mem_after = memory_snapshot();
    let index_size_mb = std::fs::metadata(&args.out)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0);

    let throughput = if embedding_ms > 0 {
        total_chunks as f64 / (embedding_ms as f64 / 1000.0)
    } else {
        0.0
    };

    let report = BuildReport {
        generated_at_unix: chrono::Utc::now().timestamp(),
        generated_at: chrono::Utc::now().to_rfc3339(),
        repo_path: args.repo.to_string_lossy().to_string(),
        model_path: args.model.to_string_lossy().to_string(),
        batch_size: args.batch,
        index_out: args.out.to_string_lossy().to_string(),
        total_files,
        total_chunks,
        timing_ms: TimingMs {
            chunking: chunking_ms,
            model_load: model_load_ms,
            embedding: embedding_ms,
            index_build: index_build_ms,
            index_save: index_save_ms,
            total: total_ms,
        },
        throughput_chunks_per_sec: throughput,
        memory_mb: MemoryReport {
            before_rss: mem_before.rss_mb,
            after_rss: mem_after.rss_mb,
            delta: mem_after.rss_mb - mem_before.rss_mb,
            peak_working_set: mem_after.peak_ws_mb,
        },
        index_size_mb,
    };

    report.print_console();

    if !args.report.is_empty() {
        report
            .write_json(&args.report)
            .with_context(|| format!("Failed to write JSON report to {}.json", args.report))?;
        report
            .write_markdown(&args.report)
            .with_context(|| format!("Failed to write MD report to {}.md", args.report))?;
        info!(
            "Report written to {}.json and {}.md",
            args.report, args.report
        );
    }

    Ok(())
}

// ─────────────────────────── search ──────────────────────────────────────────

fn cmd_search(args: SearchArgs) -> Result<()> {
    // Load index.
    let idx = Index::load(&args.index)
        .with_context(|| format!("Failed to load index from {:?}", args.index))?;

    // Load embedder (need tokenizer + model to embed the query).
    let mut embedder = Embedder::load(&args.model, &args.ort)?;

    // Embed the query.
    let query_vecs = embedder.embed_batch(&[args.query.as_str()])?;
    let query_vec = &query_vecs[0];

    // Search.
    let results = idx.search(query_vec, args.top);

    println!(
        "\n=== Top {} results for: {:?} ===\n",
        results.len(),
        args.query
    );

    for (rank, result) in results.iter().enumerate() {
        let e = &result.entry;
        let preview: String = e
            .text
            .lines()
            .take(3)
            .map(|l| format!("    | {}", l))
            .collect::<Vec<_>>()
            .join("\n");

        println!(
            "[{}] score={:.4}  {}  L{}–{}",
            rank + 1,
            result.score,
            e.file.display(),
            e.start_line,
            e.end_line
        );
        println!("{}", preview);
        println!();
    }

    Ok(())
}
