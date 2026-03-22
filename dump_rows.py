import sys
import lancedb

def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "index.db"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    db = lancedb.connect(db_path)
    chunks_table = db.open_table("chunks")

    df = chunks_table.to_pandas().head(k)

    for _, row in df.iterrows():
        print(f"--- Chunk {row['id']} ---")
        print(f"File: {row['file_path']}")
        print(f"Lines: {row['start_line']}-{row['end_line']}")
        text = row['text']
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        print(f"Text (first 200 chars): {str(text)[:200]}...")

        embedding = row['embedding']
        if embedding is not None and len(embedding) > 0:
            vec = list(embedding)[:10]
            print(f"Vector dim: {len(embedding)}")
            print(f"Vector (first 10): {vec}")
        else:
            print(f"Vector: (empty or invalid)")
        print()

if __name__ == "__main__":
    main()