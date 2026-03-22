//! SQLite-backed vector store with incremental update support.
//!
//! Schema
//! ──────
//! file_meta(path TEXT PK, content_hash TEXT, mtime_secs INTEGER)
//!   — one row per indexed source file; used to detect changes between builds.
//!
//! chunks(id INTEGER PK, file_path TEXT, start_line INTEGER, end_line INTEGER, text TEXT)
//!   — one row per chunk extracted from a source file.
//!
//! vec_chunks(chunk_id INTEGER, embedding FLOAT32 BLOB)  ← sqlite-vec virtual table
//!   — parallel to `chunks`; stores the raw embedding vector.
//!
//! Incremental build algorithm (called by main::cmd_build)
//! ────────────────────────────────────────────────────────
//! 1. Walk repo, compute sha256 of each .ets file.
//! 2. Compare with file_meta:
//!    - new file      → chunk + embed + insert
//!    - modified file → delete old chunks/vectors, re-chunk + embed + insert
//!    - deleted file  → delete old chunks/vectors, remove file_meta row
//!    - unchanged     → skip entirely
//! 3. Return list of (chunk_text, chunk_meta) that need embedding so the caller
//!    can batch them through the ONNX model and then call `insert_chunk`.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use sha2::{Digest, Sha256};

/// One chunk record ready to be persisted.
pub struct ChunkRecord {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

/// A search result returned by `Store::search`.
pub struct SearchResult {
    pub score: f32,
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

/// What happened to a file during the change-detection scan.
#[derive(Debug, PartialEq)]
pub enum FileStatus {
    New,
    Modified,
    Unchanged,
    Deleted,
}

/// Per-file change information.
pub struct FileChange {
    pub path: PathBuf,
    pub status: FileStatus,
    pub hash: String, // empty for Deleted
    pub mtime: u64,   // 0 for Deleted
}

// ─────────────────────────────────────────────────────────────────────────────

pub struct Store {
    conn: Connection,
    pub dim: usize,
}

impl Store {
    /// Open (or create) the database at `db_path`.
    /// `dim` is the embedding dimension; must match any existing data.
    /// Pass `dim = 0` when opening an existing DB for search-only use.
    pub fn open(db_path: &Path, dim: usize) -> Result<Self> {
        // Register sqlite-vec BEFORE opening the connection so the auto-extension
        // is active when the new connection runs its initialization.
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }

        let conn = Connection::open(db_path)
            .with_context(|| format!("Cannot open database {:?}", db_path))?;

        // Create schema if not present.
        conn.execute_batch(&format!(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous  = NORMAL;

            CREATE TABLE IF NOT EXISTS file_meta (
                path         TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                mtime_secs   INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path  TEXT    NOT NULL,
                start_line INTEGER NOT NULL,
                end_line   INTEGER NOT NULL,
                text       TEXT    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);

            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks
                USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[{dim}]);
            ",
        ))
        .context("Failed to initialise database schema")?;

        Ok(Self { conn, dim })
    }

    // ── Change detection ─────────────────────────────────────────────────────

    /// Scan `repo_root` for `.ets` files and compare against the stored
    /// `file_meta` table.  Returns the list of changes (new / modified /
    /// deleted).  Unchanged files are omitted.
    pub fn detect_changes(&self, repo_root: &Path) -> Result<Vec<FileChange>> {
        use walkdir::WalkDir;

        // Collect all .ets files on disk.
        let mut on_disk: std::collections::HashMap<String, (String, u64)> =
            std::collections::HashMap::new();

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
                .replace('\\', "/"); // normalise to forward slashes
            let bytes = std::fs::read(p).with_context(|| format!("Cannot read {:?}", p))?;
            let hash = hex_sha256(&bytes);
            let mtime = mtime_secs(p).unwrap_or(0);
            on_disk.insert(rel, (hash, mtime));
        }

        // Fetch what is stored.
        let mut stored: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        {
            let mut stmt = self
                .conn
                .prepare("SELECT path, content_hash FROM file_meta")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            for row in rows {
                let (path, hash) = row?;
                stored.insert(path, hash);
            }
        }

        let mut changes = Vec::new();

        // New or modified.
        for (rel, (hash, mtime)) in &on_disk {
            let status = match stored.get(rel) {
                None => FileStatus::New,
                Some(old_hash) if old_hash != hash => FileStatus::Modified,
                _ => FileStatus::Unchanged,
            };
            if status != FileStatus::Unchanged {
                changes.push(FileChange {
                    path: repo_root.join(rel),
                    status,
                    hash: hash.clone(),
                    mtime: *mtime,
                });
            }
        }

        // Deleted.
        for rel in stored.keys() {
            if !on_disk.contains_key(rel) {
                changes.push(FileChange {
                    path: repo_root.join(rel),
                    status: FileStatus::Deleted,
                    hash: String::new(),
                    mtime: 0,
                });
            }
        }

        Ok(changes)
    }

    // ── Write operations ─────────────────────────────────────────────────────

    /// Remove all chunks and the file_meta row for `rel_path`
    /// (relative path string as stored in the DB).
    pub fn delete_file(&self, rel_path: &str) -> Result<()> {
        // Delete vectors first (FK constraint order).
        self.conn.execute(
            "DELETE FROM vec_chunks WHERE chunk_id IN
             (SELECT id FROM chunks WHERE file_path = ?1)",
            params![rel_path],
        )?;
        self.conn
            .execute("DELETE FROM chunks WHERE file_path = ?1", params![rel_path])?;
        self.conn
            .execute("DELETE FROM file_meta WHERE path = ?1", params![rel_path])?;
        Ok(())
    }

    /// Insert one chunk + its embedding vector.
    pub fn insert_chunk(&self, rec: &ChunkRecord, embedding: &[f32]) -> Result<()> {
        self.conn.execute(
            "INSERT INTO chunks (file_path, start_line, end_line, text)
             VALUES (?1, ?2, ?3, ?4)",
            params![rec.file_path, rec.start_line, rec.end_line, rec.text],
        )?;
        let chunk_id = self.conn.last_insert_rowid();

        // Serialize f32 slice as little-endian bytes for sqlite-vec.
        let blob = f32_slice_to_bytes(embedding);
        self.conn.execute(
            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)",
            params![chunk_id, blob],
        )?;
        Ok(())
    }

    /// Upsert the file_meta row for `rel_path`.
    pub fn upsert_file_meta(&self, rel_path: &str, hash: &str, mtime: u64) -> Result<()> {
        self.conn.execute(
            "INSERT INTO file_meta (path, content_hash, mtime_secs)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(path) DO UPDATE SET
               content_hash = excluded.content_hash,
               mtime_secs   = excluded.mtime_secs",
            params![rel_path, hash, mtime as i64],
        )?;
        Ok(())
    }

    /// Wipe all chunks, vectors and file_meta (used by --force).
    pub fn clear(&mut self) -> Result<()> {
        self.conn
            .execute_batch("DELETE FROM vec_chunks; DELETE FROM chunks; DELETE FROM file_meta;")?;
        Ok(())
    }

    /// Wrap multiple writes in a single transaction for speed.
    pub fn transaction<F>(&mut self, f: F) -> Result<()>
    where
        F: FnOnce(&Connection) -> Result<()>,
    {
        let tx = self.conn.transaction()?;
        f(&tx)?;
        tx.commit()?;
        Ok(())
    }

    // ── Read operations ──────────────────────────────────────────────────────

    /// Return the top-`k` chunks closest to `query_vec` (cosine similarity).
    pub fn search(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        let blob = f32_slice_to_bytes(query_vec);

        let mut stmt = self.conn.prepare(
            "SELECT c.file_path, c.start_line, c.end_line, c.text,
                    v.distance
             FROM vec_chunks v
             JOIN chunks c ON c.id = v.chunk_id
             WHERE v.embedding MATCH ?1
               AND k = ?2
             ORDER BY v.distance",
        )?;

        let results = stmt
            .query_map(params![blob, top_k as i64], |row| {
                Ok(SearchResult {
                    file_path: row.get(0)?,
                    start_line: row.get::<_, i64>(1)? as usize,
                    end_line: row.get::<_, i64>(2)? as usize,
                    text: row.get(3)?,
                    // sqlite-vec returns L2 distance; convert to cosine similarity.
                    // For normalised vectors: cosine_sim = 1 - (L2² / 2).
                    score: {
                        let dist: f64 = row.get(4)?;
                        (1.0 - dist * dist / 2.0) as f32
                    },
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()
            .context("Search query failed")?;

        Ok(results)
    }

    /// Count of indexed chunks.
    pub fn chunk_count(&self) -> Result<usize> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
        Ok(n as usize)
    }

    /// Count of indexed files.
    pub fn file_count(&self) -> Result<usize> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM file_meta", [], |row| row.get(0))?;
        Ok(n as usize)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn hex_sha256(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

fn mtime_secs(path: &Path) -> Option<u64> {
    std::fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs())
}

/// Serialize an f32 slice to raw little-endian bytes (sqlite-vec wire format).
fn f32_slice_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(v.len() * 4);
    for &x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    buf
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn open_mem(dim: usize) -> Store {
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(&format!(
            "
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS file_meta (
                path TEXT PRIMARY KEY, content_hash TEXT NOT NULL, mtime_secs INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL, start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL, text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks
                USING vec0(chunk_id INTEGER PRIMARY KEY, embedding float[{dim}]);
            ",
        ))
        .unwrap();
        Store { conn, dim }
    }

    fn l2_norm(v: &[f32]) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn test_insert_and_search() {
        let store = open_mem(3);

        let vecs: &[(&str, [f32; 3])] = &[
            ("vec_a", [1.0, 0.0, 0.0]),
            ("vec_b", [0.0, 1.0, 0.0]),
            ("vec_c", [0.0, 0.0, 1.0]),
        ];

        for (text, v) in vecs {
            let norm = l2_norm(v);
            store
                .insert_chunk(
                    &ChunkRecord {
                        file_path: "test.ets".into(),
                        start_line: 1,
                        end_line: 5,
                        text: text.to_string(),
                    },
                    &norm,
                )
                .unwrap();
        }

        // Query close to vec_b.
        let query = l2_norm(&[0.05, 0.99, 0.05]);
        let results = store.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].text, "vec_b");
        assert!(results[0].score > results[2].score);
    }

    #[test]
    fn test_delete_file() {
        let store = open_mem(2);
        let v = l2_norm(&[1.0, 0.5]);
        store
            .insert_chunk(
                &ChunkRecord {
                    file_path: "a.ets".into(),
                    start_line: 1,
                    end_line: 3,
                    text: "hello".into(),
                },
                &v,
            )
            .unwrap();
        store.upsert_file_meta("a.ets", "abc123", 0).unwrap();

        assert_eq!(store.chunk_count().unwrap(), 1);
        store.delete_file("a.ets").unwrap();
        assert_eq!(store.chunk_count().unwrap(), 0);
        assert_eq!(store.file_count().unwrap(), 0);
    }
}
