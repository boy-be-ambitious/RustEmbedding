//! Performance instrumentation: wall-clock timers, Windows memory snapshots,
//! and JSON / Markdown report generation.

use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};

// ─────────────────────────── Timer ───────────────────────────────────────────

/// A simple wall-clock stopwatch.
pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }
}

// ─────────────────────────── Memory ──────────────────────────────────────────

/// A snapshot of the current process memory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemSnapshot {
    /// Resident Set Size in MiB (Linux / Windows WorkingSetSize).
    pub rss_mb: f64,
    /// Peak Working Set in MiB (Windows only; 0 elsewhere).
    pub peak_ws_mb: f64,
}

/// Take a memory snapshot of the current process.
pub fn memory_snapshot() -> MemSnapshot {
    #[cfg(windows)]
    {
        windows_memory()
    }
    #[cfg(not(windows))]
    {
        MemSnapshot::default()
    }
}

#[cfg(windows)]
fn windows_memory() -> MemSnapshot {
    use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows::Win32::System::Threading::GetCurrentProcess;

    let mut pmc = PROCESS_MEMORY_COUNTERS::default();
    let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;

    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, size) };

    if ok.is_ok() {
        const MB: f64 = 1024.0 * 1024.0;
        MemSnapshot {
            rss_mb: pmc.WorkingSetSize as f64 / MB,
            peak_ws_mb: pmc.PeakWorkingSetSize as f64 / MB,
        }
    } else {
        MemSnapshot::default()
    }
}

// ─────────────────────────── Report ──────────────────────────────────────────

/// All timing values collected during an index build, in milliseconds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMs {
    pub chunking: u64,
    pub model_load: u64,
    pub embedding: u64,
    pub index_build: u64,
    pub index_save: u64,
    pub total: u64,
}

/// The full build report, written to JSON (and optionally Markdown).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildReport {
    pub generated_at_unix: i64,
    pub generated_at: String,
    pub repo_path: String,
    pub model_path: String,
    pub batch_size: usize,
    pub index_out: String,
    pub total_files: usize,
    pub total_chunks: usize,
    pub timing_ms: TimingMs,
    pub throughput_chunks_per_sec: f64,
    pub memory_mb: MemoryReport,
    pub index_size_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReport {
    pub before_rss: f64,
    pub after_rss: f64,
    pub delta: f64,
    pub peak_working_set: f64,
}

impl BuildReport {
    /// Write the report as JSON to `<stem>.json`.
    pub fn write_json(&self, stem: &str) -> Result<()> {
        let path = format!("{}.json", stem);
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    /// Write a companion Markdown table to `<stem>.md`.
    pub fn write_markdown(&self, stem: &str) -> Result<()> {
        let path = format!("{}.md", stem);
        let md = self.to_markdown();
        std::fs::write(&path, md)?;
        Ok(())
    }

    fn to_markdown(&self) -> String {
        let mut s = String::new();
        s.push_str("# Build Report\n\n");
        s.push_str(&format!("Generated: {}\n\n", self.generated_at));

        s.push_str("## Summary\n\n");
        s.push_str("| Metric | Value |\n|---|---|\n");
        s.push_str(&format!("| Repo | `{}` |\n", self.repo_path));
        s.push_str(&format!("| Model | `{}` |\n", self.model_path));
        s.push_str(&format!("| Batch size | {} |\n", self.batch_size));
        s.push_str(&format!("| Files processed | {} |\n", self.total_files));
        s.push_str(&format!("| Chunks created | {} |\n", self.total_chunks));
        s.push_str(&format!(
            "| Throughput | {:.1} chunks/s |\n",
            self.throughput_chunks_per_sec
        ));
        s.push_str(&format!(
            "| Index file size | {:.2} MB |\n",
            self.index_size_mb
        ));

        s.push_str("\n## Timing\n\n");
        s.push_str("| Phase | ms |\n|---|---|\n");
        s.push_str(&format!("| Chunking | {} |\n", self.timing_ms.chunking));
        s.push_str(&format!("| Model load | {} |\n", self.timing_ms.model_load));
        s.push_str(&format!("| Embedding | {} |\n", self.timing_ms.embedding));
        s.push_str(&format!(
            "| Index build | {} |\n",
            self.timing_ms.index_build
        ));
        s.push_str(&format!("| Index save | {} |\n", self.timing_ms.index_save));
        s.push_str(&format!("| **Total** | **{}** |\n", self.timing_ms.total));

        s.push_str("\n## Memory\n\n");
        s.push_str("| Metric | MB |\n|---|---|\n");
        s.push_str(&format!(
            "| RSS before | {:.1} |\n",
            self.memory_mb.before_rss
        ));
        s.push_str(&format!(
            "| RSS after | {:.1} |\n",
            self.memory_mb.after_rss
        ));
        s.push_str(&format!("| Delta | {:.1} |\n", self.memory_mb.delta));
        s.push_str(&format!(
            "| Peak working set | {:.1} |\n",
            self.memory_mb.peak_working_set
        ));

        s
    }

    /// Print the console banner (mirrors the README sample output).
    pub fn print_console(&self) {
        println!();
        println!("====== Build Stats ======");
        println!("  Repo              : {}", self.repo_path);
        println!("  Files processed   : {}", self.total_files);
        println!("  Chunks created    : {}", self.total_chunks);
        println!("  Batch size        : {}", self.batch_size);
        println!("--------------------------");
        println!("  Chunking time     : {}ms", self.timing_ms.chunking);
        println!("  Model load time   : {}ms", self.timing_ms.model_load);
        println!("  Embedding time    : {}ms", self.timing_ms.embedding);
        println!("  Index build time  : {}ms", self.timing_ms.index_build);
        println!("  Index save time   : {}ms", self.timing_ms.index_save);
        println!("  TOTAL             : {}ms", self.timing_ms.total);
        println!(
            "  Throughput        : {:.1} chunks/s",
            self.throughput_chunks_per_sec
        );
        println!("--------------------------");
        println!("  Mem before (RSS)  : {:.1} MB", self.memory_mb.before_rss);
        println!("  Mem after  (RSS)  : {:.1} MB", self.memory_mb.after_rss);
        println!("  Mem delta         : {:.1} MB", self.memory_mb.delta);
        println!(
            "  Peak working set  : {:.1} MB",
            self.memory_mb.peak_working_set
        );
        println!("--------------------------");
        println!("  Index file size   : {:.2} MB", self.index_size_mb);
        println!("=========================");
    }
}
