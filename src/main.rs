mod chunker;
mod embedder;
mod embedder_concurrent;
mod index;
mod perf;
mod store_lance;
mod tokenizer;

use std::path::PathBuf;
use std::time::{Instant, SystemTime};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use log::info;
use walkdir::WalkDir;
use sha2::{Digest, Sha256};

use crate::embedder::Embedder;
use crate::perf::{IndexProfiler, Timer};
use crate::store_lance::{ChunkRecord, FileStatus, LanceStore};

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
    Build(BuildArgs),
    Search(SearchArgs),
    Benchmark(BenchmarkArgs),
}

#[derive(Parser, Debug)]
struct BuildArgs {
    #[arg(long, default_value = "hmosworld-master")]
    repo: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    ort: PathBuf,
    #[arg(long, default_value = "index.lance")]
    db: PathBuf,
    #[arg(long, default_value_t = 16)]
    batch: usize,
    #[arg(long, default_value = "")]
    report: String,
    #[arg(long, default_value_t = false)]
    force: bool,
}

#[derive(Parser, Debug)]
struct SearchArgs {
    #[arg(long)]
    query: String,
    #[arg(long, default_value = "index.lance")]
    db: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    ort: PathBuf,
    #[arg(long, default_value_t = 5)]
    top: usize,
}

#[derive(Parser, Debug)]
struct BenchmarkArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    ort: PathBuf,
    #[arg(long, default_value_t = 100)]
    num_texts: usize,
    #[arg(long, default_value_t = 4)]
    num_threads: usize,
}

fn main() -> Result<()> {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    let cli = Cli::parse();
    match cli.command {
        Command::Build(a) => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(cmd_build(a))
        }
        Command::Search(a) => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(cmd_search(a))
        }
        Command::Benchmark(a) => cmd_benchmark(a),
    }
}

fn hex_sha256(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

fn mtime_secs(path: &std::path::Path) -> Option<u64> {
    std::fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}

fn rel_str(path: &std::path::Path, base: &std::path::Path) -> String {
    path.strip_prefix(base).unwrap_or(path).to_string_lossy().replace('\\', "/")
}

async fn cmd_build(args: BuildArgs) -> Result<()> {
    let profiler = IndexProfiler::start(&args.db);

    info!("Loading tokenizer and ONNX model ...");
    let model_timer = Timer::start();
    let mut embedder = Embedder::load(&args.model, &args.ort)?;
    info!("  Model loaded ({:.2}s).", model_timer.elapsed_secs());

    let store = LanceStore::open(&args.db, embedder.dim).await?;
    store.create_table_if_not_exists().await?;
    info!("Database: {:?}", args.db);
    info!("Scanning {:?} for changes ...", args.repo);

    if args.force {
        info!("  --force: clearing existing index.");
        store.clear().await?;
    }

    let changes = detect_changes(&args.repo, &store).await?;

    let new_count = changes.iter().filter(|c| c.status == FileStatus::New).count();
    let mod_count = changes.iter().filter(|c| c.status == FileStatus::Modified).count();
    let del_count = changes.iter().filter(|c| c.status == FileStatus::Deleted).count();
    info!("  Changes: {} new, {} modified, {} deleted", new_count, mod_count, del_count);

    if changes.is_empty() {
        info!("Nothing to do — index is up to date.");
        let _ = profiler.stop(0.0, 0.0, 0, 0);
        return Ok(());
    }

    for ch in changes.iter().filter(|c| c.status == FileStatus::Deleted || c.status == FileStatus::Modified) {
        let rel = rel_str(&ch.path, &args.repo);
        store.delete_by_file(&rel).await?;
        if ch.status == FileStatus::Deleted {
            info!("  deleted  {}", rel);
        }
    }

    let to_embed: Vec<_> = changes.iter().filter(|c| c.status == FileStatus::New || c.status == FileStatus::Modified).collect();

    let chunk_timer = Timer::start();
    let mut all_chunks: Vec<ChunkRecord> = Vec::new();
    for ch in to_embed.iter() {
        let source = std::fs::read_to_string(&ch.path).with_context(|| format!("Cannot read {:?}", ch.path))?;
        let rel = rel_str(&ch.path, &args.repo);
        for fc in chunker::chunk_source(&source, PathBuf::from(&rel)) {
            all_chunks.push(ChunkRecord {
                file_path: rel.clone(),
                start_line: fc.start_line,
                end_line: fc.end_line,
                text: fc.text,
                embedding: vec![],
            });
        }
    }
    info!("  {} files → {} chunks to embed ({:.0}ms chunking)", to_embed.len(), all_chunks.len(), chunk_timer.elapsed_ms() as f64);

    info!("Embedding {} chunks (batch_size={}) ...", all_chunks.len(), args.batch);
    let embed_timer = Timer::start();
    let total_chunks = all_chunks.len();
    let mut done = 0usize;

    for batch_start in (0..total_chunks).step_by(args.batch) {
        let batch_end = (batch_start + args.batch).min(total_chunks);
        let texts: Vec<&str> = all_chunks[batch_start..batch_end].iter().map(|c| c.text.as_str()).collect();
        let vecs = embedder.embed_batch(&texts)?;
        for (i, vec) in vecs.into_iter().enumerate() {
            all_chunks[batch_start + i].embedding = vec;
        }
        done += batch_end - batch_start;
        if done % 50 < args.batch || done == total_chunks {
            info!("  {}/{} chunks embedded", done, total_chunks);
        }
    }
    let embedding_secs = embed_timer.elapsed_secs();
    info!("  Embedding done ({:.2}s)", embedding_secs);

    let db_write_timer = Timer::start();
    store.insert_chunks(&all_chunks).await?;
    
    for ch in &to_embed {
        let rel = rel_str(&ch.path, &args.repo);
        store.upsert_file_meta(&rel, &ch.hash, ch.mtime).await?;
    }
    let db_write_secs = db_write_timer.elapsed_secs();

    let file_count = store.file_count().await?;
    let chunk_count = store.chunk_count().await?;
    let metrics = profiler.stop(embedding_secs, db_write_secs, file_count, chunk_count);

    metrics.print_console();

    if !args.report.is_empty() {
        metrics.write_json(&args.report)?;
        metrics.write_markdown(&args.report)?;
        info!("Report written to {}.json and {}.md", args.report, args.report);
    }

    Ok(())
}

async fn cmd_search(args: SearchArgs) -> Result<()> {
    let store = LanceStore::open(&args.db, 0).await?;
    let mut embedder = Embedder::load(&args.model, &args.ort)?;

    let query_vecs = embedder.embed_batch(&[args.query.as_str()])?;
    let results = store.search(&query_vecs[0], args.top).await?;

    println!("\n=== Top {} results for: {:?} ===\n", results.len(), args.query);

    for (rank, r) in results.iter().enumerate() {
        let preview: String = r.text.lines().take(3).map(|l| format!("    | {}", l)).collect::<Vec<_>>().join("\n");
        println!("[{}] score={:.4}  {}  L{}–{}", rank + 1, r.score, r.file_path, r.start_line, r.end_line);
        println!("{}", preview);
        println!();
    }

    Ok(())
}

fn cmd_benchmark(args: BenchmarkArgs) -> Result<()> {
    println!("=== Concurrency Benchmark ===");
    println!("Model: {:?}", args.model);
    println!("ORT: {:?}", args.ort);
    println!("Num texts: {}", args.num_texts);
    println!("Num threads: {}", args.num_threads);
    println!();

    let embedder = crate::embedder_concurrent::ConcurrentEmbedder::load(&args.model, &args.ort)?;

    let texts: Vec<String> = (0..args.num_texts)
        .map(|i| format!("function test_{}() {{ return {}; }}", i, i))
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    println!("Running sequential embedding...");
    let start = Instant::now();
    let _results_seq = embedder.embed_batch(&text_refs)?;
    let seq_duration = start.elapsed();
    println!("Sequential: {:?}", seq_duration);

    println!("\nRunning parallel embedding ({} threads)...", args.num_threads);
    let start = Instant::now();
    let _results_par = embedder.embed_batch_parallel(&text_refs, args.num_threads)?;
    let par_duration = start.elapsed();
    println!("Parallel: {:?}", par_duration);

    let speedup = seq_duration.as_secs_f64() / par_duration.as_secs_f64();
    println!("\n=== Results ===");
    println!("Speedup: {:.2}x", speedup);
    println!("Throughput (seq): {:.2} texts/sec", args.num_texts as f64 / seq_duration.as_secs_f64());
    println!("Throughput (par): {:.2} texts/sec", args.num_texts as f64 / par_duration.as_secs_f64());

    Ok(())
}

async fn detect_changes(repo_root: &std::path::Path, store: &LanceStore) -> Result<Vec<crate::store_lance::FileChange>> {
    let mut on_disk: std::collections::HashMap<String, (String, u64)> = std::collections::HashMap::new();

    for entry in WalkDir::new(repo_root)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let p = entry.path();
        if p.extension().and_then(|s| s.to_str()) != Some("ets") {
            continue;
        }
        let rel = p
            .strip_prefix(repo_root)
            .unwrap_or(p)
            .to_string_lossy()
            .replace('\\', "/");
        let bytes = std::fs::read(p).with_context(|| format!("Cannot read {:?}", p))?;
        let hash = hex_sha256(&bytes);
        let mtime = mtime_secs(p).unwrap_or(0);
        on_disk.insert(rel, (hash, mtime));
    }

    let stored = store.get_all_file_meta().await?;

    let mut changes = Vec::new();

    for (rel, (hash, mtime)) in &on_disk {
        let status = match stored.get(rel) {
            None => FileStatus::New,
            Some(old_hash) if old_hash != hash => FileStatus::Modified,
            _ => FileStatus::Unchanged,
        };
        if status != FileStatus::Unchanged {
            changes.push(crate::store_lance::FileChange {
                path: repo_root.join(rel),
                status,
                hash: hash.clone(),
                mtime: *mtime,
            });
        }
    }

    for rel in stored.keys() {
        if !on_disk.contains_key(rel) {
            changes.push(crate::store_lance::FileChange {
                path: repo_root.join(rel),
                status: FileStatus::Deleted,
                hash: String::new(),
                mtime: 0,
            });
        }
    }

    Ok(changes)
}
