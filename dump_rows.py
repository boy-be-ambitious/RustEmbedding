import sqlite3
import struct
import sys
import sqlite_vec

def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "index.db"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    cur = conn.cursor()
    
    cur.execute("SELECT id, file_path, start_line, end_line, text FROM chunks LIMIT ?", (k,))
    chunks = cur.fetchall()
    
    for chunk_id, file_path, start_line, end_line, text in chunks:
        cur.execute("SELECT embedding FROM vec_chunks WHERE chunk_id = ?", (chunk_id,))
        row = cur.fetchone()
        if row:
            blob = row[0]
            vec = struct.unpack(f'<{len(blob)//4}f', blob)
            
            print(f"--- Chunk {chunk_id} ---")
            print(f"File: {file_path}")
            print(f"Lines: {start_line}-{end_line}")
            print(f"Text (first 200 chars): {text[:200]}...")
            print(f"Vector dim: {len(vec)}")
            print(f"Vector (first 10): {vec[:10]}")
            print()

    conn.close()

if __name__ == "__main__":
    main()
