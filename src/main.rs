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
use crate::perf::{IndexProfiler, Timer};
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
    // ── 1. Start profiler (baseline snapshot + background sampler) ────────────
    let profiler = IndexProfiler::start(&args.db);

    // ── 2. Load model ─────────────────────────────────────────────────────────
    info!("Loading tokenizer and ONNX model ...");
    let model_timer = Timer::start();
    let mut embedder = Embedder::load(&args.model, &args.ort)?;
    info!("  Model loaded ({:.2}s).", model_timer.elapsed_secs());

    // ── 3. Open DB and detect changes ─────────────────────────────────────────
    let mut store = Store::open(&args.db, embedder.dim)?;
    info!("Database: {:?}", args.db);
    info!("Scanning {:?} for changes ...", args.repo);

    let changes = if args.force {
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
        new_count, mod_count, del_count
    );

    if changes.is_empty() {
        info!("Nothing to do — index is up to date.");
        // Stop profiler but skip writing report (nothing was embedded).
        let _ = profiler.stop(0.0, 0.0, store.file_count()?, store.chunk_count()?);
        return Ok(());
    }

    // ── 4. Apply deletions ────────────────────────────────────────────────────
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

    // ── 5. Chunk files that need (re-)embedding ────────────────────────────────
    let to_embed: Vec<_> = changes
        .iter()
        .filter(|c| c.status == FileStatus::New || c.status == FileStatus::Modified)
        .collect();

    let chunk_timer = Timer::start();
    let mut all_chunks: Vec<ChunkRecord> = Vec::new();
    for ch in to_embed.iter() {
        let source = std::fs::read_to_string(&ch.path)
            .with_context(|| format!("Cannot read {:?}", ch.path))?;
        let rel = rel_str(&ch.path, &args.repo);
        for fc in chunker::chunk_source(&source, PathBuf::from(&rel)) {
            all_chunks.push(ChunkRecord {
                file_path: rel.clone(),
                start_line: fc.start_line,
                end_line: fc.end_line,
                text: fc.text,
            });
        }
    }
    info!(
        "  {} files → {} chunks to embed ({:.0}ms chunking)",
        to_embed.len(),
        all_chunks.len(),
        chunk_timer.elapsed_ms() as f64
    );

    // ── 6. Embed in batches ───────────────────────────────────────────────────
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
    let embedding_secs = embed_timer.elapsed_secs();
    info!("  Embedding done ({:.2}s)", embedding_secs);

    // ── 7. Update file_meta ───────────────────────────────────────────────────
    let db_write_timer = Timer::start();
    for ch in &to_embed {
        let rel = rel_str(&ch.path, &args.repo);
        store.upsert_file_meta(&rel, &ch.hash, ch.mtime)?;
    }
    let db_write_secs = db_write_timer.elapsed_secs();

    // ── 8. Stop profiler and emit report ──────────────────────────────────────
    let file_count = store.file_count()?;
    let chunk_count = store.chunk_count()?;
    let metrics = profiler.stop(embedding_secs, db_write_secs, file_count, chunk_count);

    metrics.print_console();

    if !args.report.is_empty() {
        metrics.write_json(&args.report)?;
        metrics.write_markdown(&args.report)?;
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
