use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use lancedb::prelude::*;
use lance::dataset::Dataset;
use arrow_array::{RecordBatch, StringArray, Int64Array, Float32Array, ArrayRef};
use arrow_schema::{Schema, Field, DataType};
use sha2::{Digest, Sha256};

pub struct ChunkRecord {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub embedding: Vec<f32>,
}

pub struct SearchResult {
    pub score: f32,
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

#[derive(Debug, PartialEq)]
pub enum FileStatus {
    New,
    Modified,
    Unchanged,
    Deleted,
}

pub struct FileChange {
    pub path: PathBuf,
    pub status: FileStatus,
    pub hash: String,
    pub mtime: u64,
}

pub struct LanceStore {
    db: Connection,
    table_name: String,
    uri: String,
    pub dim: usize,
}

impl LanceStore {
    pub async fn open(db_path: &Path, dim: usize) -> Result<Self> {
        let uri = db_path.to_string_lossy().to_string();
        let db = connect(&uri).execute().await
            .context("Failed to connect to LanceDB")?;
        
        let table_name = "chunks".to_string();
        
        Ok(Self {
            db,
            table_name,
            uri,
            dim,
        })
    }

    pub async fn create_table_if_not_exists(&self) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("file_path", DataType::Utf8, false),
            Field::new("start_line", DataType::Int64, false),
            Field::new("end_line", DataType::Int64, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("embedding", DataType::Float32, false),
        ]));

        let tables = self.db.table_names().execute().await?;
        if !tables.contains(&self.table_name) {
            let empty_batch = RecordBatch::new_empty(schema);
            self.db
                .create_table(&self.table_name)
                .add(Box::new(std::iter::once(empty_batch)))
                .execute()
                .await?;
        }

        Ok(())
    }

    pub async fn insert_chunks(&self, chunks: &[ChunkRecord]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let mut ids = Vec::with_capacity(chunks.len());
        let mut file_paths = Vec::with_capacity(chunks.len());
        let mut start_lines = Vec::with_capacity(chunks.len());
        let mut end_lines = Vec::with_capacity(chunks.len());
        let mut texts = Vec::with_capacity(chunks.len());
        let mut embeddings = Vec::with_capacity(chunks.len());

        for (i, chunk) in chunks.iter().enumerate() {
            ids.push(i as i64);
            file_paths.push(chunk.file_path.clone());
            start_lines.push(chunk.start_line as i64);
            end_lines.push(chunk.end_line as i64);
            texts.push(chunk.text.clone());
            embeddings.extend_from_slice(&chunk.embedding);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("file_path", DataType::Utf8, false),
            Field::new("start_line", DataType::Int64, false),
            Field::new("end_line", DataType::Int64, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("embedding", DataType::Float32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(ids)) as ArrayRef,
                Arc::new(StringArray::from(file_paths)) as ArrayRef,
                Arc::new(Int64Array::from(start_lines)) as ArrayRef,
                Arc::new(Int64Array::from(end_lines)) as ArrayRef,
                Arc::new(StringArray::from(texts)) as ArrayRef,
                Arc::new(Float32Array::from(embeddings)) as ArrayRef,
            ],
        )?;

        self.db
            .open_table(&self.table_name)
            .execute()
            .await?
            .add(Box::new(std::iter::once(Ok(batch))))
            .execute()
            .await?;

        Ok(())
    }

    pub async fn search(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        let table = self.db.open_table(&self.table_name).execute().await?;
        
        let mut results = table
            .vector_search(query_vec.to_vec())
            .column("embedding")
            .nprobes(20)
            .refine_factor(10)
            .build()?
            .limit(top_k)
            .execute()
            .await?;

        let mut search_results = Vec::new();
        
        while let Some(batch) = results.try_next().await? {
            let file_paths = batch.column_by_name("file_path")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())?;
            let start_lines = batch.column_by_name("start_line")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>())?;
            let end_lines = batch.column_by_name("end_line")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>())?;
            let texts = batch.column_by_name("text")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())?;
            let distances = batch.column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            for i in 0..batch.num_rows() {
                let score = if let Some(d) = distances {
                    1.0 - d.value(i)
                } else {
                    1.0
                };
                
                search_results.push(SearchResult {
                    score,
                    file_path: file_paths.value(i).to_string(),
                    start_line: start_lines.value(i) as usize,
                    end_line: end_lines.value(i) as usize,
                    text: texts.value(i).to_string(),
                });
            }
        }

        Ok(search_results)
    }

    pub async fn delete_by_file(&self, file_path: &str) -> Result<()> {
        let table = self.db.open_table(&self.table_name).execute().await?;
        table
            .delete(&format!("file_path = '{}'", file_path))
            .execute()
            .await?;
        Ok(())
    }

    pub async fn clear(&self) -> Result<()> {
        self.db.drop_table(&self.table_name).execute().await?;
        self.create_table_if_not_exists().await?;
        Ok(())
    }
}

fn hex_sha256(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_lance_store_insert_and_search() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_lance");
        
        let store = LanceStore::open(&db_path, 3).await.unwrap();
        store.create_table_if_not_exists().await.unwrap();

        let chunks = vec![
            ChunkRecord {
                file_path: "test.ets".to_string(),
                start_line: 1,
                end_line: 5,
                text: "hello world".to_string(),
                embedding: vec![1.0, 0.0, 0.0],
            },
            ChunkRecord {
                file_path: "test.ets".to_string(),
                start_line: 6,
                end_line: 10,
                text: "foo bar".to_string(),
                embedding: vec![0.0, 1.0, 0.0],
            },
        ];

        store.insert_chunks(&chunks).await.unwrap();

        let query = vec![1.0, 0.1, 0.1];
        let results = store.search(&query, 2).await.unwrap();
        
        assert!(!results.is_empty());
    }
}
