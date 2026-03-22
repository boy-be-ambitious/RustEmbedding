//! Regex-based chunker for ArkTS / HarmonyOS `.ets` files.
//!
//! Splits a source file at structural boundaries:
//!   - top-level `@Component` / `@Entry` struct declarations
//!   - top-level `function` / `async function` declarations
//!   - top-level `class` declarations
//!   - top-level `export` statements that introduce the above
//!
//! Any text between two boundaries becomes one chunk; the very first chunk
//! (the file header / imports) is kept if it is non-empty.

use std::path::{Path, PathBuf};

use anyhow::Result;
use walkdir::WalkDir;

/// A single chunk extracted from a source file.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Path to the source file (relative to the repo root).
    pub file: PathBuf,
    /// 1-based line number where the chunk starts.
    pub start_line: usize,
    /// 1-based line number where the chunk ends (inclusive).
    pub end_line: usize,
    /// The raw source text of the chunk.
    pub text: String,
}

/// Walk `repo_root` recursively, collect every `.ets` file, split each file
/// into chunks, and return all chunks in order.
pub fn chunk_repo(repo_root: &Path) -> Result<Vec<Chunk>> {
    let mut all: Vec<Chunk> = Vec::new();

    for entry in WalkDir::new(repo_root)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("ets") {
            let source = std::fs::read_to_string(path)?;
            let rel = path.strip_prefix(repo_root).unwrap_or(path).to_path_buf();
            let chunks = chunk_source(&source, rel);
            all.extend(chunks);
        }
    }

    Ok(all)
}

/// Split a single source `text` into chunks, tagging each with `file`.
pub fn chunk_source(text: &str, file: PathBuf) -> Vec<Chunk> {
    // Detect boundary lines: lines that start a top-level declaration.
    // We treat a line as a boundary when it (possibly preceded by decorators)
    // begins a struct, class, or function declaration.
    let lines: Vec<&str> = text.lines().collect();
    let n = lines.len();

    // Collect the 0-based indices of lines that are "boundary starters".
    let mut boundaries: Vec<usize> = Vec::new();

    let mut i = 0;
    while i < n {
        let trimmed = lines[i].trim();

        if is_boundary_line(trimmed) {
            // Walk backwards to include any decorator lines (@Component etc.)
            // that immediately precede this line.
            let start = preceding_decorator_start(&lines, i);
            boundaries.push(start);
            i += 1;
            continue;
        }
        i += 1;
    }

    // Deduplicate and sort (preceding_decorator_start can introduce overlaps).
    boundaries.sort_unstable();
    boundaries.dedup();

    if boundaries.is_empty() {
        // The whole file is one chunk.
        if text.trim().is_empty() {
            return vec![];
        }
        return vec![Chunk {
            file,
            start_line: 1,
            end_line: n,
            text: text.to_owned(),
        }];
    }

    let mut chunks: Vec<Chunk> = Vec::new();

    // First chunk: lines before the first boundary (header / imports).
    if boundaries[0] > 0 {
        let header: String = lines[..boundaries[0]].join("\n");
        if !header.trim().is_empty() {
            chunks.push(Chunk {
                file: file.clone(),
                start_line: 1,
                end_line: boundaries[0],
                text: header,
            });
        }
    }

    // One chunk per boundary segment.
    for w in boundaries.windows(2) {
        let (s, e) = (w[0], w[1]);
        let seg: String = lines[s..e].join("\n");
        if !seg.trim().is_empty() {
            chunks.push(Chunk {
                file: file.clone(),
                start_line: s + 1,
                end_line: e,
                text: seg,
            });
        }
    }

    // Last segment: from the last boundary to end of file.
    let last = *boundaries.last().unwrap();
    let tail: String = lines[last..].join("\n");
    if !tail.trim().is_empty() {
        chunks.push(Chunk {
            file: file.clone(),
            start_line: last + 1,
            end_line: n,
            text: tail,
        });
    }

    chunks
}

/// Returns `true` when `line` looks like the start of a top-level declaration.
fn is_boundary_line(line: &str) -> bool {
    // Strip leading `export` keyword.
    let rest = line
        .strip_prefix("export default ")
        .or_else(|| line.strip_prefix("export "))
        .unwrap_or(line);

    rest.starts_with("function ")
        || rest.starts_with("async function ")
        || rest.starts_with("class ")
        || rest.starts_with("abstract class ")
        || rest.starts_with("struct ")
        || rest.starts_with("interface ")
        || rest.starts_with("enum ")
        || rest.starts_with("type ")
        || rest.starts_with("const ")
        || rest.starts_with("let ")
}

/// Walk backwards from `idx` to include any decorator lines (`@…`) that
/// immediately precede the declaration line.
fn preceding_decorator_start(lines: &[&str], idx: usize) -> usize {
    let mut start = idx;
    while start > 0 {
        let prev = lines[start - 1].trim();
        if prev.starts_with('@') || prev.is_empty() {
            start -= 1;
        } else {
            break;
        }
    }
    start
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"import { media } from '@ohos.multimedia.media';
import { common } from '@ohos.app.ability.common';

@Component
export struct AudioPlayer {
  private player: media.AVPlayer | null = null;
  build() {
    Column() { Text("Play") }
  }
}

@Entry
@Component
export struct HomePage {
  build() {
    Column() { Text("Home") }
  }
}

export function helper(x: number): number {
  return x * 2;
}

export class AudioService {
  init() {}
}
"#;

    #[test]
    fn test_chunk_count() {
        let chunks = chunk_source(SAMPLE, PathBuf::from("test.ets"));
        // Expect: header/imports + AudioPlayer + HomePage + helper + AudioService = 5
        assert!(
            chunks.len() >= 4,
            "expected >= 4 chunks, got {}",
            chunks.len()
        );
    }

    #[test]
    fn test_decorator_included_in_chunk() {
        let chunks = chunk_source(SAMPLE, PathBuf::from("test.ets"));
        let audio = chunks
            .iter()
            .find(|c| c.text.contains("AudioPlayer"))
            .expect("AudioPlayer chunk not found");
        assert!(
            audio.text.contains("@Component"),
            "decorator should be included in chunk"
        );
    }

    #[test]
    fn test_line_numbers() {
        let chunks = chunk_source(SAMPLE, PathBuf::from("test.ets"));
        for chunk in &chunks {
            assert!(chunk.start_line <= chunk.end_line);
            assert!(chunk.start_line >= 1);
        }
    }

    #[test]
    fn test_empty_file() {
        let chunks = chunk_source("", PathBuf::from("empty.ets"));
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_no_boundaries() {
        // A source with no structural declarations is one chunk.
        let src = "// just a comment\n// another comment\n";
        let chunks = chunk_source(src, PathBuf::from("simple.ets"));
        assert_eq!(chunks.len(), 1);
    }
}
