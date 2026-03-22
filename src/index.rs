//! Flat vector index with cosine-similarity retrieval.
//!
//! All vectors are stored as `Vec<f32>` in a plain `Vec<Entry>`.  Because each
//! entry's vector is L2-normalised by the embedder, cosine similarity reduces
//! to a dot product, which is cheap to compute.
//!
//! Serialisation uses `bincode` (v1 API) for a compact binary representation.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// A single indexed entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    /// Relative path of the source file.
    pub file: PathBuf,
    /// 1-based start line.
    pub start_line: usize,
    /// 1-based end line (inclusive).
    pub end_line: usize,
    /// The raw source text (used for display in search results).
    pub text: String,
    /// L2-normalised embedding vector.
    pub vector: Vec<f32>,
}

/// The full flat index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index {
    entries: Vec<Entry>,
    /// Dimensionality of the stored vectors.
    pub dim: usize,
}

/// A search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub score: f32,
    pub entry: Entry,
}

impl Index {
    /// Create an empty index with a known vector dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            entries: Vec::new(),
            dim,
        }
    }

    /// Add an entry to the index.
    pub fn add(&mut self, entry: Entry) {
        self.entries.push(entry);
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialise to a binary file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let bytes = bincode::serialize(self).context("Failed to serialise index")?;
        std::fs::write(path, bytes)
            .with_context(|| format!("Failed to write index to {:?}", path))?;
        Ok(())
    }

    /// Deserialise from a binary file.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("Failed to read index from {:?}", path))?;
        let index: Self = bincode::deserialize(&bytes).context("Failed to deserialise index")?;
        Ok(index)
    }

    /// Return the top-`k` entries ranked by cosine similarity to `query_vec`.
    /// Assumes all stored vectors are already L2-normalised (dot product == cosine sim).
    pub fn search(&self, query_vec: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut scored: Vec<(f32, usize)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (dot(query_vec, &e.vector), i))
            .collect();

        // Partial sort — bring the top_k largest scores to the front.
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(top_k)
            .map(|(score, idx)| SearchResult {
                score,
                entry: self.entries[idx].clone(),
            })
            .collect()
    }
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_entry(vector: Vec<f32>, text: &str) -> Entry {
        Entry {
            file: PathBuf::from("test.ets"),
            start_line: 1,
            end_line: 5,
            text: text.to_owned(),
            vector,
        }
    }

    fn l2_norm(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_ranking_order() {
        let mut idx = Index::new(3);
        // Three unit vectors.
        idx.add(make_entry(l2_norm(&[1.0, 0.0, 0.0]), "vec_a"));
        idx.add(make_entry(l2_norm(&[0.0, 1.0, 0.0]), "vec_b"));
        idx.add(make_entry(l2_norm(&[0.0, 0.0, 1.0]), "vec_c"));

        // Query close to vec_b (strongly weighted).
        let query = l2_norm(&[0.05, 0.99, 0.05]);
        let results = idx.search(&query, 3);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].entry.text, "vec_b");
        // vec_a and vec_c have the same cosine similarity to this query,
        // so just assert top result is clearly better than third.
        assert!(results[0].score > results[2].score);
    }

    #[test]
    fn test_bincode_roundtrip() {
        let mut idx = Index::new(2);
        idx.add(make_entry(l2_norm(&[1.0, 0.5]), "hello world"));
        idx.add(make_entry(l2_norm(&[0.2, 0.8]), "second entry"));

        let bytes = bincode::serialize(&idx).unwrap();
        let decoded: Index = bincode::deserialize(&bytes).unwrap();

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded.entries[1].text, "second entry");
    }

    #[test]
    fn test_top_k_clamps() {
        let mut idx = Index::new(2);
        for i in 0..5 {
            idx.add(make_entry(
                l2_norm(&[i as f32 + 1.0, 0.5]),
                &format!("e{}", i),
            ));
        }
        let query = l2_norm(&[1.0, 0.0]);
        let results = idx.search(&query, 3);
        assert_eq!(results.len(), 3);
    }
}
