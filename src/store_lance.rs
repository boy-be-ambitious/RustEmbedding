use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use lancedb::prelude::*;
use arrow_array::{RecordBatch, StringArray, Int64Array, Float32Array, ArrayRef, UInt8Array, FixedSizeListArray};
use arrow_schema::{Schema, Field, DataType};
use sha2::{Digest, Sha256};
use futures::StreamExt;

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

#[derive(Debug, PartialEq, Clone)]
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
    file_meta_table: String,
    pub dim: usize,
}

impl LanceStore {
    pub async fn open(db_path: &Path, dim: usize) -> Result<Self> {
        let uri = db_path.to_string_lossy().to_string();
        let db = connect(&uri).execute().await
            .context("Failed to connect to LanceDB")?;
        
        Ok(Self {
            db,
            table_name: "chunks".to_string(),
            file_meta_table: "file_meta".to_string(),
            dim,
        })
    }

    pub async fn create_table_if_not_exists(&self) -> Result<()> {
        let tables = self.db.table_names().execute().await?;
        
        if !tables.contains(&self.table_name) {
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("file_path", DataType::Utf8, false),
                Field::new("start_line", DataType::Int64, false),
                Field::new("end_line", DataType::Int64, false),
                Field::new("text", DataType::Utf8, false),
                Field::new("embedding", DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    768
                ), false),
            ]));
            let empty_batch = RecordBatch::new_empty(schema);
            self.db
                .create_table(&self.table_name)
                .add(Box::new(std::iter::once(Ok(empty_batch))))
                .execute()
                .await?;
        }

        if !tables.contains(&self.file_meta_table) {
            let schema = Arc::new(Schema::new(vec![
                Field::new("path", DataType::Utf8, false),
                Field::new("content_hash", DataType::Utf8, false),
                Field::new("mtime_secs", DataType::Int64, false),
            ]));
            let empty_batch = RecordBatch::new_empty(schema);
            self.db
                .create_table(&self.file_meta_table)
                .add(Box::new(std::iter::once(Ok(empty_batch))))
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
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());

        for (i, chunk) in chunks.iter().enumerate() {
            ids.push(i as i64);
            file_paths.push(chunk.file_path.clone());
            start_lines.push(chunk.start_line as i64);
            end_lines.push(chunk.end_line as i64);
            texts.push(chunk.text.clone());
            embeddings.push(chunk.embedding.clone());
        }

        let embedding_arr = FixedSizeListArray::try_new_from_values(
            Float32Array::from(embeddings.into_iter().flatten().collect::<Vec<f32>>()),
            768
        )?;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("file_path", DataType::Utf8, false),
            Field::new("start_line", DataType::Int64, false),
            Field::new("end_line", DataType::Int64, false),
            Field::new("text", DataType::Utf8, false),
            Field::new("embedding", DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                768
            ), false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(ids)) as ArrayRef,
                Arc::new(StringArray::from(file_paths)) as ArrayRef,
                Arc::new(Int64Array::from(start_lines)) as ArrayRef,
                Arc::new(Int64Array::from(end_lines)) as ArrayRef,
                Arc::new(StringArray::from(texts)) as ArrayRef,
                Arc::new(embedding_arr) as ArrayRef,
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
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let start_lines = batch.column_by_name("start_line")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
            let end_lines = batch.column_by_name("end_line")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
            let texts = batch.column_by_name("text")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let distances = batch.column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            if let (Some(fp), Some(sl), Some(el), Some(txt)) = (file_paths, start_lines, end_lines, texts) {
                for i in 0..batch.num_rows() {
                    let score = if let Some(d) = distances {
                        1.0 - d.value(i)
                    } else {
                        1.0
                    };
                    
                    search_results.push(SearchResult {
                        score,
                        file_path: fp.value(i).to_string(),
                        start_line: sl.value(i) as usize,
                        end_line: el.value(i) as usize,
                        text: txt.value(i).to_string(),
                    });
                }
            }
        }

        Ok(search_results)
    }

    pub async fn delete_by_file(&self, file_path: &str) -> Result<()> {
        let table = self.db.open_table(&self.table_name).execute().await?;
        table
            .delete(&format!("file_path = '{}'", file_path.replace("'", "''")))
            .execute()
            .await?;
        
        let meta_table = self.db.open_table(&self.file_meta_table).execute().await?;
        meta_table
            .delete(&format!("path = '{}'", file_path.replace("'", "''")))
            .execute()
            .await?;
        
        Ok(())
    }

    pub async fn upsert_file_meta(&self, path: &str, hash: &str, mtime: u64) -> Result<()> {
        let meta_table = self.db.open_table(&self.file_meta_table).execute().await?;
        meta_table
            .delete(&format!("path = '{}'", path.replace("'", "''")))
            .execute()
            .await?;

        let schema = Arc::new(Schema::new(vec![
            Field::new("path", DataType::Utf8, false),
            Field::new("content_hash", DataType::Utf8, false),
            Field::new("mtime_secs", DataType::Int64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![path])) as ArrayRef,
                Arc::new(StringArray::from(vec![hash])) as ArrayRef,
                Arc::new(Int64Array::from(vec![mtime as i64])) as ArrayRef,
            ],
        )?;

        self.db
            .open_table(&self.file_meta_table)
            .execute()
            .await?
            .add(Box::new(std::iter::once(Ok(batch))))
            .execute()
            .await?;

        Ok(())
    }

    pub async fn get_all_file_meta(&self) -> Result<HashMap<String, String>> {
        let mut result = HashMap::new();
        let table = self.db.open_table(&self.file_meta_table).execute().await?;
        let mut scanner = table.scan();
        let mut stream = scanner.try_into_stream().await?;

        while let Some(batch) = stream.try_next().await? {
            let paths = batch.column_by_name("path")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let hashes = batch.column_by_name("content_hash")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            if let (Some(p), Some(h)) = (paths, hashes) {
                for i in 0..batch.num_rows() {
                    result.insert(p.value(i).to_string(), h.value(i).to_string());
                }
            }
        }

        Ok(result)
    }

    pub async fn clear(&self) -> Result<()> {
        self.db.drop_table(&self.table_name).execute().await?;
        self.db.drop_table(&self.file_meta_table).execute().await?;
        self.create_table_if_not_exists().await?;
        Ok(())
    }

    pub async fn chunk_count(&self) -> Result<usize> {
        let table = self.db.open_table(&self.table_name).execute().await?;
        let count = table.count_rows(None).await?;
        Ok(count)
    }

    pub async fn file_count(&self) -> Result<usize> {
        let table = self.db.open_table(&self.file_meta_table).execute().await?;
        let count = table.count_rows(None).await?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_lance_store_insert_and_search() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_lance");
        
        let store = LanceStore::open(&db_path, 768).await.unwrap();
        store.create_table_if_not_exists().await.unwrap();

        let chunks = vec![
            ChunkRecord {
                file_path: "test.ets".to_string(),
                start_line: 1,
                end_line: 5,
                text: "hello world".to_string(),
                embedding: vec![1.0; 768],
            },
        ];

        store.insert_chunks(&chunks).await.unwrap();

        let query = vec![1.0; 768];
        let results = store.search(&query, 1).await.unwrap();
        
        assert_eq!(results.len(), 1);
    }
}
