//! CLI entry point for RustEmbedding.
//!
//! Sub-commands:
//!   build  – incrementally chunk/embed a repo into a SQLite database
//!   search – semantic search against an existing database

mod chunker;
mod embedder;
mod index; // kept for unit tests
mod perf;
mod store;
mod tokenizer;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use log::info;

use crate::embedder::Embedder;
use crate::perf::{memory_snapshot, BuildReport, MemoryReport, Timer, TimingMs};
use crate::store::{ChunkRecord, FileStatus, Store};

// ─────────────────────────── CLI ─────────────────────────────────────────────

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
    /// Incrementally build / update the vector index.
    Build(BuildArgs),
    /// Query the index.
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

    /// SQLite database file (created if absent).
    #[arg(long, default_value = "index.db")]
    db: PathBuf,

    /// ONNX inference batch size (16–32 recommended).
    #[arg(long, default_value_t = 16)]
    batch: usize,

    /// Stem for report files; writes <stem>.json and <stem>.md.
    /// Leave empty to skip.
    #[arg(long, default_value = "")]
    report: String,

    /// Force full re-index even if files are unchanged.
    #[arg(long, default_value_t = false)]
    force: bool,
}

#[derive(Parser, Debug)]
struct SearchArgs {
    /// Natural-language or code query.
    #[arg(long)]
    query: String,

    /// SQLite database produced by `build`.
    #[arg(long, default_value = "index.db")]
    db: PathBuf,

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
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    let cli = Cli::parse();
    match cli.command {
        Command::Build(a) => cmd_build(a),
        Command::Search(a) => cmd_search(a),
    }
}

// ─────────────────────────── build ───────────────────────────────────────────

fn cmd_build(args: BuildArgs) -> Result<()> {
    let total_timer = Timer::start();
    let mem_before = memory_snapshot();

    // ── 1. Load model ─────────────────────────────────────────────────────────
    info!("Loading tokenizer and ONNX model ...");
    let model_timer = Timer::start();
    let mut embedder = Embedder::load(&args.model, &args.ort)?;
    let model_load_ms = model_timer.elapsed_ms();
    info!("  Model loaded ({:.2}s).", model_load_ms as f64 / 1000.0);

    // ── 2. Open DB and detect changes ─────────────────────────────────────────
    let mut store = Store::open(&args.db, embedder.dim)?;
    info!("Database: {:?}", args.db);

    let chunk_timer = Timer::start();
    info!("Scanning {:?} for changes ...", args.repo);

    let changes = if args.force {
        // Treat every file as new by clearing the DB first.
        info!("  --force: clearing existing index.");
        store.clear()?;
        store.detect_changes(&args.repo)?
    } else {
        store.detect_changes(&args.repo)?
    };

    let new_count = changes
        .iter()
        .filter(|c| c.status == FileStatus::New)
        .count();
    let mod_count = changes
        .iter()
        .filter(|c| c.status == FileStatus::Modified)
        .count();
    let del_count = changes
        .iter()
        .filter(|c| c.status == FileStatus::Deleted)
        .count();
    info!(
        "  Changes: {} new, {} modified, {} deleted",
        new_count, mod_count, del_count,
    );

    if changes.is_empty() {
        info!("Nothing to do — index is up to date.");
        let total_ms = total_timer.elapsed_ms();
        info!("TOTAL: {}ms", total_ms);
        return Ok(());
    }

    // ── 3. Apply deletions ────────────────────────────────────────────────────
    for ch in changes
        .iter()
        .filter(|c| c.status == FileStatus::Deleted || c.status == FileStatus::Modified)
    {
        let rel = rel_str(&ch.path, &args.repo);
        store.delete_file(&rel)?;
        if ch.status == FileStatus::Deleted {
            info!("  deleted  {}", rel);
        }
    }

    // ── 4. Chunk files that need (re-)embedding ────────────────────────────────
    let to_embed: Vec<_> = changes
        .iter()
        .filter(|c| c.status == FileStatus::New || c.status == FileStatus::Modified)
        .collect();

    // Collect all chunks across all files to embed in one batched pass.
    let mut all_chunks: Vec<ChunkRecord> = Vec::new();
    // Map chunk index → (file change index) for writing file_meta later.
    let mut chunk_file_map: Vec<usize> = Vec::new();

    for (fi, ch) in to_embed.iter().enumerate() {
        let source = std::fs::read_to_string(&ch.path)
            .with_context(|| format!("Cannot read {:?}", ch.path))?;
        let rel = rel_str(&ch.path, &args.repo);
        let file_chunks = chunker::chunk_source(&source, PathBuf::from(&rel));
        for fc in file_chunks {
            all_chunks.push(ChunkRecord {
                file_path: rel.clone(),
                start_line: fc.start_line,
                end_line: fc.end_line,
                text: fc.text,
            });
            chunk_file_map.push(fi);
        }
    }

    let chunking_ms = chunk_timer.elapsed_ms();
    info!(
        "  {} files → {} chunks to embed ({:.2}ms chunking)",
        to_embed.len(),
        all_chunks.len(),
        chunking_ms as f64
    );

    // ── 5. Embed in batches ───────────────────────────────────────────────────
    info!(
        "Embedding {} chunks (batch_size={}) ...",
        all_chunks.len(),
        args.batch
    );
    let embed_timer = Timer::start();
    let total_chunks = all_chunks.len();

    let mut done = 0usize;
    for batch_start in (0..total_chunks).step_by(args.batch) {
        let batch_end = (batch_start + args.batch).min(total_chunks);
        let texts: Vec<&str> = all_chunks[batch_start..batch_end]
            .iter()
            .map(|c| c.text.as_str())
            .collect();
        let vecs = embedder.embed_batch(&texts)?;

        for (i, vec) in vecs.into_iter().enumerate() {
            store.insert_chunk(&all_chunks[batch_start + i], &vec)?;
        }

        done += batch_end - batch_start;
        if done % 50 < args.batch || done == total_chunks {
            info!("  {}/{} chunks embedded", done, total_chunks);
        }
    }

    let embedding_ms = embed_timer.elapsed_ms();
    info!("  Embedding done ({:.2}s)", embedding_ms as f64 / 1000.0);

    // ── 6. Update file_meta for new/modified files ────────────────────────────
    let save_timer = Timer::start();
    for ch in &to_embed {
        let rel = rel_str(&ch.path, &args.repo);
        store.upsert_file_meta(&rel, &ch.hash, ch.mtime)?;
    }
    let index_save_ms = save_timer.elapsed_ms();

    // ── 7. Stats ──────────────────────────────────────────────────────────────
    let total_ms = total_timer.elapsed_ms();
    let mem_after = memory_snapshot();
    let db_size_mb = std::fs::metadata(&args.db)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0);
    let total_files = store.file_count()?;
    let total_stored_chunks = store.chunk_count()?;
    let throughput = if embedding_ms > 0 {
        total_chunks as f64 / (embedding_ms as f64 / 1000.0)
    } else {
        0.0
    };

    let report = BuildReport {
        generated_at_unix: chrono::Utc::now().timestamp(),
        generated_at: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        repo_path: args.repo.to_string_lossy().to_string(),
        model_path: args.model.to_string_lossy().to_string(),
        batch_size: args.batch,
        index_out: args.db.to_string_lossy().to_string(),
        total_files,
        total_chunks: total_stored_chunks,
        timing_ms: TimingMs {
            chunking: chunking_ms,
            model_load: model_load_ms,
            embedding: embedding_ms,
            index_build: 0,
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
        index_size_mb: db_size_mb,
    };

    report.print_console();

    if !args.report.is_empty() {
        report.write_json(&args.report)?;
        report.write_markdown(&args.report)?;
        info!(
            "Report written to {}.json and {}.md",
            args.report, args.report
        );
    }

    Ok(())
}

// ─────────────────────────── search ──────────────────────────────────────────

fn cmd_search(args: SearchArgs) -> Result<()> {
    let store = Store::open(&args.db, 0)?; // dim=0: store reads it from DB schema
    let mut embedder = Embedder::load(&args.model, &args.ort)?;

    let query_vecs = embedder.embed_batch(&[args.query.as_str()])?;
    let results = store.search(&query_vecs[0], args.top)?;

    println!(
        "\n=== Top {} results for: {:?} ===\n",
        results.len(),
        args.query
    );

    for (rank, r) in results.iter().enumerate() {
        let preview: String = r
            .text
            .lines()
            .take(3)
            .map(|l| format!("    | {}", l))
            .collect::<Vec<_>>()
            .join("\n");
        println!(
            "[{}] score={:.4}  {}  L{}–{}",
            rank + 1,
            r.score,
            r.file_path,
            r.start_line,
            r.end_line
        );
        println!("{}", preview);
        println!();
    }

    Ok(())
}

// ─────────────────────────── helpers ─────────────────────────────────────────

/// Return the path relative to `base` as a forward-slash string.
fn rel_str(path: &std::path::Path, base: &std::path::Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}
