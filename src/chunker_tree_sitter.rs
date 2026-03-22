use anyhow::Result;
use std::path::PathBuf;
use tree_sitter::{Node, Parser};

pub struct Chunk {
    pub file: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub node_type: String,
}

pub struct TreeSitterChunker {
    parser: Parser,
}

impl TreeSitterChunker {
    pub fn new() -> Result<Self> {
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_typescript::language_typescript())?;
        Ok(Self { parser })
    }

    pub fn chunk_source(&self, source: &str, file: PathBuf) -> Vec<Chunk> {
        let tree = self.parser.parse(source, None);
        match tree {
            Some(t) => {
                let root = t.root_node();
                self.extract_chunks(root, source, file)
            }
            None => vec![],
        }
    }

    fn extract_chunks(&self, node: Node, source: &str, file: PathBuf) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let mut cursor = node.walk();

        if cursor.goto_first_child() {
            loop {
                let child = cursor.node();
                let node_type = child.kind();

                if Self::is_structural_node(node_type) {
                    if let Some(chunk) = self.extract_structural_chunk(&child, source, file.clone())
                    {
                        chunks.push(chunk);
                    }
                }

                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }

        chunks
    }

    fn is_structural_node(node_type: &str) -> bool {
        matches!(
            node_type,
            "struct_declaration"
                | "class_declaration"
                | "interface_declaration"
                | "enum_declaration"
                | "type_alias_declaration"
                | "function_declaration"
                | "method_definition"
                | "export_statement"
                | "lexical_declaration"
                | "variable_declaration"
        )
    }

    fn extract_structural_chunk(&self, node: &Node, source: &str, file: PathBuf) -> Option<Chunk> {
        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        let start = node.start_position().row;
        let end = node.end_position().row;

        if start_byte >= end_byte || end_byte > source.len() {
            return None;
        }

        let text = source[start_byte..end_byte].to_string();
        let node_type = node.kind().to_string();

        Some(Chunk {
            file,
            start_line: start + 1,
            end_line: end + 1,
            text,
            node_type,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_sitter_chunker() {
        let chunker = TreeSitterChunker::new().unwrap();
        let source = r#"
import { media } from '@ohos.multimedia.media';

@Component
export struct AudioPlayer {
  private player: media.AVPlayer | null = null;
  build() {
    Column() { Text("Play") }
  }
}

export function helper(x: number): number {
  return x * 2;
}
"#;
        let chunks = chunker.chunk_source(source, PathBuf::from("test.ets"));
        assert!(
            chunks.len() >= 1,
            "Expected at least 1 chunk, got {}",
            chunks.len()
        );
    }
}
