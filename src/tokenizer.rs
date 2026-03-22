//! HuggingFace tokenizer wrapper.
//!
//! Loads a `tokenizer.json` from disk, then exposes a `tokenize` method that
//! returns `(input_ids, attention_mask, token_type_ids)` tensors ready for the
//! VESO-25M ONNX model.
//!
//! Padding is applied to the *batch maximum length*, capped at 512 tokens.

use anyhow::Result;
use tokenizers::Tokenizer;

pub struct EtsTokenizer {
    inner: Tokenizer,
}

/// Output of a single tokenisation call.
pub struct Encoding {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub token_type_ids: Vec<i64>,
}

impl EtsTokenizer {
    /// Load a `tokenizer.json` from `model_dir`.
    pub fn load(model_dir: &std::path::Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {:?}: {}", path, e))?;
        Ok(Self { inner: tokenizer })
    }

    /// Tokenize a batch of strings. Returns a flat row-major matrix together
    /// with the sequence length used (padded to batch-max, capped at 512).
    ///
    /// Returns `(input_ids, attention_mask, token_type_ids, seq_len)`
    pub fn encode_batch(&self, texts: &[&str]) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>, usize)> {
        const MAX_LEN: usize = 512;

        // Encode without padding first to find the batch-max length.
        let mut raw: Vec<tokenizers::Encoding> = Vec::with_capacity(texts.len());
        for &t in texts {
            let enc = self
                .inner
                .encode(t, false)
                .map_err(|e| anyhow::anyhow!("Tokenizer encode error: {}", e))?;
            raw.push(enc);
        }

        let seq_len = raw
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(MAX_LEN);
        let seq_len = seq_len.max(1); // avoid 0-length tensors

        let batch = texts.len();
        let mut input_ids = vec![0i64; batch * seq_len];
        let mut attention_mask = vec![0i64; batch * seq_len];
        let mut token_type_ids = vec![0i64; batch * seq_len];

        for (i, enc) in raw.iter().enumerate() {
            let ids = enc.get_ids();
            let ttype = enc.get_type_ids();
            let len = ids.len().min(seq_len);
            let offset = i * seq_len;
            for j in 0..len {
                input_ids[offset + j] = ids[j] as i64;
                attention_mask[offset + j] = 1;
                token_type_ids[offset + j] = ttype[j] as i64;
            }
        }

        Ok((input_ids, attention_mask, token_type_ids, seq_len))
    }

    /// Convenience wrapper: encode a single string and return its token count.
    pub fn token_count(&self, text: &str) -> Result<usize> {
        let enc = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenizer encode error: {}", e))?;
        Ok(enc.get_ids().len())
    }
}

#[cfg(test)]
mod tests {
    // Unit tests for tokenizer require the model directory to exist,
    // so they are integration-style and skipped in CI.
    // The chunker and index tests are fully self-contained.
}
