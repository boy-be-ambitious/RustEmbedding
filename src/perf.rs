//! Performance instrumentation: wall-clock timers, background resource sampler,
//! and JSON / Markdown report generation.
//!
//! # Design (mirrors examplProfiler.rs)
//!
//! `IndexProfiler::start(db_path)` captures baseline memory/CPU, records the
//! start storage size, then spawns a background thread that samples the process
//! every 500 ms and tracks peak memory (MB) and peak CPU (%).
//!
//! The main thread calls `profiler.stop(…)` when the build finishes, which
//! signals the sampler thread to exit and returns a filled `IndexMetrics`.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use sysinfo::{Pid, ProcessesToUpdate, System};

// ─────────────────────────── Timer ───────────────────────────────────────────

/// Simple wall-clock stopwatch.
pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

// ─────────────────────────── IndexMetrics ────────────────────────────────────

/// The report written to `<stem>.json` and printed to the console.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetrics {
    /// Total wall-clock seconds for the full build.
    pub total_time: f64,
    /// Seconds spent on ONNX embedding (file-level chunks).
    pub embedding_file_time: f64,
    /// Seconds spent writing file-level chunks to the DB.
    pub db_write_file_time: f64,
    /// Reserved — symbol-level embedding (always 0).
    pub embedding_symbol_time: f64,
    /// Reserved — symbol-level DB write (always 0).
    pub db_write_symbol_time: f64,
    /// Process RSS at build start in MB.
    pub start_memory: f64,
    /// Peak process RSS observed during the build in MB.
    pub peak_memory: f64,
    /// Process CPU% at build start.
    pub start_cpu: f64,
    /// Peak process CPU% observed during the build.
    pub peak_cpu: f64,
    /// DB file size in MB before the build.
    pub start_storage: f64,
    /// DB file size in MB after the build.
    pub end_storage: f64,
    /// Total source files indexed.
    pub file_count: usize,
    /// Total chunks stored.
    pub chunk_count: usize,
}

impl IndexMetrics {
    /// Write as pretty-printed JSON to `<stem>.json`.
    pub fn write_json(&self, stem: &str) -> Result<()> {
        std::fs::write(
            format!("{}.json", stem),
            serde_json::to_string_pretty(self)?,
        )?;
        Ok(())
    }

    /// Write a Markdown table to `<stem>.md`.
    pub fn write_markdown(&self, stem: &str) -> Result<()> {
        std::fs::write(format!("{}.md", stem), self.to_markdown())?;
        Ok(())
    }

    fn to_markdown(&self) -> String {
        let mut s = String::new();
        s.push_str("# Build Report\n\n");
        s.push_str("| Metric | Value |\n|---|---|\n");
        s.push_str(&format!("| total_time | {:.2}s |\n", self.total_time));
        s.push_str(&format!(
            "| embedding_file_time | {:.2}s |\n",
            self.embedding_file_time
        ));
        s.push_str(&format!(
            "| db_write_file_time | {:.2}s |\n",
            self.db_write_file_time
        ));
        s.push_str(&format!(
            "| embedding_symbol_time | {:.2}s |\n",
            self.embedding_symbol_time
        ));
        s.push_str(&format!(
            "| db_write_symbol_time | {:.2}s |\n",
            self.db_write_symbol_time
        ));
        s.push_str(&format!("| start_memory | {:.2} MB |\n", self.start_memory));
        s.push_str(&format!("| peak_memory | {:.2} MB |\n", self.peak_memory));
        s.push_str(&format!("| start_cpu | {:.2}% |\n", self.start_cpu));
        s.push_str(&format!("| peak_cpu | {:.2}% |\n", self.peak_cpu));
        s.push_str(&format!(
            "| start_storage | {:.2} MB |\n",
            self.start_storage
        ));
        s.push_str(&format!("| end_storage | {:.2} MB |\n", self.end_storage));
        s.push_str(&format!("| file_count | {} |\n", self.file_count));
        s.push_str(&format!("| chunk_count | {} |\n", self.chunk_count));
        s
    }

    /// Print in the `IndexMetrics { … }` console format.
    pub fn print_console(&self) {
        println!();
        println!("IndexMetrics {{");
        println!("  total_time: {:.2}s", self.total_time);
        println!("  embedding_file_time: {:.2}s", self.embedding_file_time);
        println!("  db_write_file_time: {:.2}s", self.db_write_file_time);
        println!(
            "  embedding_symbol_time: {:.2}s",
            self.embedding_symbol_time
        );
        println!("  db_write_symbol_time: {:.2}s", self.db_write_symbol_time);
        println!("  start_memory: {:.2} MB", self.start_memory);
        println!("  peak_memory: {:.2} MB", self.peak_memory);
        println!("  start_cpu: {:.2}%", self.start_cpu);
        println!("  peak_cpu: {:.2}%", self.peak_cpu);
        println!("  start_storage: {:.2} MB", self.start_storage);
        println!("  end_storage: {:.2} MB", self.end_storage);
        println!("  file_count: {}", self.file_count);
        println!("  chunk_count: {}", self.chunk_count);
        println!("}}");
    }
}

// ─────────────────────────── Sampler shared state ────────────────────────────

#[derive(Debug, Default)]
struct SamplerState {
    peak_memory_mb: f64,
    peak_cpu_pct: f64,
}

// ─────────────────────────── IndexProfiler ───────────────────────────────────

/// Lifecycle-scoped profiler that mirrors `IndexProfiler` in examplProfiler.rs.
///
/// ```
/// let profiler = IndexProfiler::start(&db_path);
/// // … do work …
/// let metrics = profiler.stop(embed_secs, db_write_secs, file_count, chunk_count);
/// metrics.print_console();
/// ```
pub struct IndexProfiler {
    start_time: Instant,
    start_memory_mb: f64,
    start_cpu_pct: f64,
    start_storage_mb: f64,
    db_path: std::path::PathBuf,
    running: Arc<AtomicBool>,
    state: Arc<Mutex<SamplerState>>,
}

impl IndexProfiler {
    /// Snapshot baseline resources, spawn the sampler thread, return `Self`.
    pub fn start(db_path: &Path) -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let pid = sysinfo::get_current_pid().expect("cannot get current pid");
        let (start_memory_mb, start_cpu_pct) = read_process(&sys, pid);
        let start_storage_mb = file_size_mb(db_path);

        let running = Arc::new(AtomicBool::new(true));
        let state = Arc::new(Mutex::new(SamplerState {
            peak_memory_mb: start_memory_mb,
            peak_cpu_pct: start_cpu_pct,
        }));

        spawn_sampler(pid, running.clone(), state.clone());

        Self {
            start_time: Instant::now(),
            start_memory_mb,
            start_cpu_pct,
            start_storage_mb,
            db_path: db_path.to_path_buf(),
            running,
            state,
        }
    }

    /// Stop the sampler thread and return the completed `IndexMetrics`.
    ///
    /// - `embedding_secs`  — seconds spent in ONNX forward passes
    /// - `db_write_secs`   — seconds spent writing chunks to SQLite
    /// - `file_count`      — total source files indexed
    /// - `chunk_count`     — total chunks stored
    pub fn stop(
        self,
        embedding_secs: f64,
        db_write_secs: f64,
        file_count: usize,
        chunk_count: usize,
    ) -> IndexMetrics {
        // Signal sampler to exit; give it one sleep interval to notice.
        self.running.store(false, Ordering::Relaxed);
        thread::sleep(Duration::from_millis(600));

        let total_time = self.start_time.elapsed().as_secs_f64();
        let end_storage_mb = file_size_mb(&self.db_path);
        let st = self.state.lock().unwrap();

        IndexMetrics {
            total_time,
            embedding_file_time: embedding_secs,
            db_write_file_time: db_write_secs,
            embedding_symbol_time: 0.0,
            db_write_symbol_time: 0.0,
            start_memory: self.start_memory_mb,
            peak_memory: st.peak_memory_mb,
            start_cpu: self.start_cpu_pct,
            peak_cpu: st.peak_cpu_pct,
            start_storage: self.start_storage_mb,
            end_storage: end_storage_mb,
            file_count,
            chunk_count,
        }
    }
}

// ─────────────────────────── private helpers ─────────────────────────────────

fn spawn_sampler(pid: Pid, running: Arc<AtomicBool>, state: Arc<Mutex<SamplerState>>) {
    thread::spawn(move || {
        let mut sys = System::new_all();
        while running.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(500));
            sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), false);
            if let Some(proc) = sys.process(pid) {
                const MB: f64 = 1024.0 * 1024.0;
                let mem_mb = proc.memory() as f64 / MB;
                let cpu_pct = proc.cpu_usage() as f64;
                let mut st = state.lock().unwrap();
                if mem_mb > st.peak_memory_mb {
                    st.peak_memory_mb = mem_mb;
                }
                if cpu_pct > st.peak_cpu_pct {
                    st.peak_cpu_pct = cpu_pct;
                }
            }
        }
    });
}

fn read_process(sys: &System, pid: Pid) -> (f64, f64) {
    if let Some(proc) = sys.process(pid) {
        const MB: f64 = 1024.0 * 1024.0;
        (proc.memory() as f64 / MB, proc.cpu_usage() as f64)
    } else {
        (0.0, 0.0)
    }
}

fn file_size_mb(path: &Path) -> f64 {
    std::fs::metadata(path)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0)
}
