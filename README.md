# RustEmbedding

Semantic code search for **ArkTS / HarmonyOS `.ets` files**.  
The tool chunks source files by structural boundaries, embeds each chunk with a
local ONNX model (VESO-25M, FP16), stores the vectors in a **SQLite + sqlite-vec
database**, and answers natural-language or code queries with cosine-similarity
retrieval.

Subsequent builds are **incremental**: only files whose `sha256` content hash has
changed are re-chunked and re-embedded; unchanged files are skipped entirely.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Rust toolchain (stable) | Install from [rustup.rs](https://rustup.rs) |
| `onnxruntime.dll` (Windows) | Use the **official CPU release**: download `onnxruntime-win-x64-1.24.4.zip` from the [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases) and extract to e.g. `C:\Users\<you>\Downloads\onnxruntime-win-x64-1.24.4`; the dll is at `lib\onnxruntime.dll` inside that directory. **Version must be 1.24.x** (the `ort` crate is built against API level 24). Bundled Lingma / DevEco dlls are incompatible (wrong version or hang on load). |
| VESO-25M model directory | Bundled with **DevEco Studio** at `C:\Program Files\Huawei\DevEco Studio\plugins\codegenie-plugin\embedding_model\VESO-model\VESO-25M`; must contain `tokenizer.json` and `model_fp16.onnx` |
| Test corpus (`.ets` files) | `hmosworld-master/` is included in this repo and is used in the examples below |

---

## Quick start (PowerShell scripts)

Three ready-to-run scripts are provided.  Edit the `$Model` / `$Ort` defaults
at the top of each file if your paths differ from the defaults.

| Script | Purpose |
|---|---|
| `build.ps1` | Compile the project, incrementally index a repo, write `report.json` / `report.md` |
| `search.ps1` | Query an existing index |
| `test.ps1` | Run the unit test suite |

---

## Step 1 — Compile and build the index

```powershell
# Defaults — uses hmosworld-master, DevEco VESO-25M, ORT 1.24.4
.\build.ps1

# Explicit paths (all defaults shown)
.\build.ps1 `
  -Repo   "hmosworld-master" `
  -Model  "C:\Program Files\Huawei\DevEco Studio\plugins\codegenie-plugin\embedding_model\VESO-model\VESO-25M" `
  -Ort    "C:\Users\qinzh\Downloads\onnxruntime-win-x64-1.24.4\lib\onnxruntime.dll" `
  -Db     "index.db" `
  -Batch  16 `
  -Report "report"
```

Default parameters (override with flags):

| Parameter | Default | Description |
|---|---|---|
| `-Repo` | `hmosworld-master` | Root directory to scan for `.ets` files |
| `-Model` | DevEco Studio VESO-25M path | Directory containing `tokenizer.json` and `model_fp16.onnx` |
| `-Ort` | `C:\Users\qinzh\Downloads\onnxruntime-win-x64-1.24.4\lib\onnxruntime.dll` | Path to `onnxruntime.dll` (must be v1.24.x) |
| `-Db` | `index.db` | SQLite database file (created on first run, updated incrementally afterwards) |
| `-Batch` | `16` | ONNX inference batch size (16–32 recommended) |
| `-Report` | `report` | Stem for report files: writes `<stem>.json` and `<stem>.md` |
| `-SkipCompile` | off | Skip `cargo build --release` and use the existing binary |
| `-Force` | off | Wipe the database and re-embed everything from scratch |

Example with a custom repo and batch size:

```powershell
.\build.ps1 `
  -Repo   "path\to\my\repo" `
  -Ort    "C:\Users\qinzh\Downloads\onnxruntime-win-x64-1.24.4\lib\onnxruntime.dll" `
  -Batch  32 `
  -Report my_report
```

> **Tip:** set `RUST_LOG` to control log verbosity.
> `RUST_LOG=info` (the default) prints progress; `RUST_LOG=warn` suppresses it.

### First run — full index

```
[INFO] Model loaded (0.29s).
[INFO] Database: "index.db"
[INFO] Scanning "hmosworld-master" for changes ...
[INFO]   Changes: 161 new, 0 modified, 0 deleted
[INFO]   161 files → 788 chunks to embed (36ms chunking)
[INFO] Embedding 788 chunks (batch_size=16) ...
[INFO]   64/788 chunks embedded
  ...
[INFO]   Embedding done (22.19s)

====== Build Stats ======
  Repo              : hmosworld-master
  Files processed   : 161
  Chunks created    : 788
  Batch size        : 16
--------------------------
  Chunking time     : 36ms
  Model load time   : 289ms
  Embedding time    : 22190ms
  Index build time  : 0ms
  Index save time   : 3ms
  TOTAL             : 22548ms
  Throughput        : 35.5 chunks/s
--------------------------
  Mem before (RSS)  : 5.1 MB
  Mem after  (RSS)  : 184.0 MB
  Mem delta         : 178.9 MB
  Peak working set  : 419.4 MB
--------------------------
  Index file size   : 3.84 MB
=========================
```

### Subsequent runs — incremental update

When no files have changed, the tool finishes in under a second:

```
[INFO] Model loaded (0.28s).
[INFO] Database: "index.db"
[INFO] Scanning "hmosworld-master" for changes ...
[INFO]   Changes: 0 new, 0 modified, 0 deleted
[INFO] Nothing to do — index is up to date.
[INFO] TOTAL: 312ms
```

When only some files change, only those are re-embedded:

```
[INFO]   Changes: 2 new, 1 modified, 0 deleted
[INFO]   3 files → 18 chunks to embed
[INFO] Embedding 18 chunks (batch_size=16) ...
[INFO]   Embedding done (0.52s)
```

### Metrics JSON (`report.json`)

```json
{
  "generated_at_unix": 1742601722,
  "generated_at": "2026-03-22T05:06:29Z",
  "repo_path": "hmosworld-master",
  "model_path": "C:\\...\\VESO-25M",
  "batch_size": 16,
  "index_out": "index.db",
  "total_files": 161,
  "total_chunks": 788,
  "timing_ms": {
    "chunking": 36,
    "model_load": 289,
    "embedding": 22190,
    "index_build": 0,
    "index_save": 3,
    "total": 22548
  },
  "throughput_chunks_per_sec": 35.5,
  "memory_mb": {
    "before_rss": 5.1,
    "after_rss": 184.0,
    "delta": 178.9,
    "peak_working_set": 419.4
  },
  "index_size_mb": 3.84
}
```

A companion `report.md` Markdown table is also written alongside the JSON.

---

## Step 2 — Query the index

```powershell
# Short form (uses index.db + defaults)
.\search.ps1 -Query "AudioPlayer initialization"
.\search.ps1 -Query "network request" -Top 10

# Explicit paths
.\search.ps1 `
  -Query  "AudioPlayer initialization" `
  -Db     "index.db" `
  -Model  "C:\Program Files\Huawei\DevEco Studio\plugins\codegenie-plugin\embedding_model\VESO-model\VESO-25M" `
  -Ort    "C:\Users\qinzh\Downloads\onnxruntime-win-x64-1.24.4\lib\onnxruntime.dll" `
  -Top    5
```

| Parameter | Default | Description |
|---|---|---|
| `-Query` | **required** | Natural-language or code query |
| `-Db` | `index.db` | SQLite database produced by `build.ps1` |
| `-Model` | DevEco Studio VESO-25M path | Model directory (tokenizer) |
| `-Ort` | `C:\Users\qinzh\Downloads\onnxruntime-win-x64-1.24.4\lib\onnxruntime.dll` | Path to `onnxruntime.dll` (must be v1.24.x) |
| `-Top` | `5` | Number of results to return |

### Example output

```
=== Top 5 results for: "AudioPlayer initialization" ===

[1] score=0.7834  commons/audioplayer/src/main/ets/service/SpeechPlayerService.ets  L72–72
    |         let filePath = path + TEMP_AUDIO_FILE_NAME;

[2] score=0.7586  features/mine/src/main/ets/service/UserNetFunc.ets  L166–166
    |         const aaid: string = await AAID.getAAID();

[3] score=0.7447  features/mine/src/main/ets/service/UserNetFunc.ets  L167–168
    |         const pushToken: string = await pushService.getToken();
    |         Logger.info(TAG, 'Get AAID successfully: %{public}s', aaid);
...
```

---

## 原理与数据链路

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        build 阶段                            │
│                                                             │
│  .ets 源文件  →  Chunker  →  Tokenizer  →  ONNX(VESO-25M)  │
│                   分块          BPE 编码      FP16 推理      │
│                                               ↓             │
│                                         masked mean-pool    │
│                                         L2 归一化           │
│                                               ↓             │
│                                    float32 向量 (768 维)    │
│                                               ↓             │
│                              SQLite + sqlite-vec (index.db) │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        search 阶段                           │
│                                                             │
│  自然语言查询  →  Tokenizer  →  ONNX  →  查询向量 (768 维)  │
│                                               ↓             │
│                              sqlite-vec KNN（余弦相似度）    │
│                                               ↓             │
│                              Top-K chunk + 文件路径 + 行号   │
└─────────────────────────────────────────────────────────────┘
```

### 各模块职责

#### 1. 分块（`chunker.rs`）

以结构边界为分割点将 `.ets` 源文件切成若干 **chunk**，每个 chunk 是一个语义独立的代码片段。

分块规则：识别以下顶层声明的起始行作为边界：

| 边界模式 | 示例 |
|---|---|
| `@Component` / `@Entry` struct | `@Component export struct AudioPlayer { ... }` |
| `function` / `async function` | `export function helper(x: number) { ... }` |
| `class` / `abstract class` | `export class AudioService { ... }` |
| `interface` / `enum` / `type` | `interface IPlayer { ... }` |

装饰器行（`@Component` 等）会被归入紧随其后的声明，确保语义完整。  
文件头部（`import` 语句等）如果非空也会保留为独立 chunk。

#### 2. Tokenization（`tokenizer.rs`）

使用 VESO-25M 自带的 `tokenizer.json`（HuggingFace BPE 格式，`onig` 正则后端）对每个 chunk 文本进行编码：

- 最大序列长度：**512 tokens**，超出截断
- Batch 内按最长序列补 padding（`input_ids = 0`，`attention_mask = 0`）
- 输出：`input_ids`、`attention_mask` 两个 int64 tensor

#### 3. ONNX 推理（`embedder.rs`）

模型：**VESO-25M FP16**（华为 DevEco Studio 内置，专为 HarmonyOS 代码设计）

推理流程：

```
输入: input_ids [batch, seq_len]  int64
      attention_mask [batch, seq_len]  int64
         ↓
  VESO-25M Transformer (FP16, CPU)
         ↓
输出: last_hidden_state [batch, seq_len, 768]  float32
         ↓
  masked mean-pool（只对 attention_mask=1 的 token 取均值）
         ↓
  L2 归一化（使向量单位化，余弦相似度 = 点积）
         ↓
  float32 向量 [batch, 768]
```

批量大小默认 16，在内存与吞吐之间取得平衡（约 35 chunk/s，CPU）。

**DLL 加载特殊处理**：ort crate 默认用 `LoadLibraryExW + LOAD_WITH_ALTERED_SEARCH_PATH` 加载 dll，在 Windows 11 启用了 Smart App Control 的环境下会触发云端 reputation 校验而永久阻塞。本项目改用 `LoadLibraryW`（无附加标志）加载 dll，再通过 `ort::set_api()` 直接注入 `OrtApi` 指针，完全绕过 libloading 的加载路径。

#### 4. 向量存储与检索（`store.rs`）

数据库结构：

```sql
-- 文件级变更追踪
file_meta(path TEXT PK, content_hash TEXT, mtime_secs INTEGER)

-- chunk 元数据（文件路径、行号、原始文本）
chunks(id INTEGER PK, file_path TEXT, start_line INTEGER,
       end_line INTEGER, text TEXT)

-- sqlite-vec 虚拟表（存储 float32 向量，支持 KNN 查询）
vec_chunks USING vec0(chunk_id INTEGER PK, embedding float[768])
```

检索时 sqlite-vec 使用 **L2 距离**做 KNN，对于 L2 归一化的向量 L2 距离与余弦相似度等价：

```
cosine_similarity = 1 - L2_distance² / 2
```

#### 5. 增量更新（`store.rs` + `main.rs`）

每次 `build` 运行时：

```
walk repo → sha256(文件内容) → 与 file_meta 比对
    │
    ├── 新文件    → chunk + embed → INSERT chunks / vec_chunks
    │             → UPSERT file_meta
    ├── 内容变更  → DELETE 旧 chunks / vec_chunks
    │             → chunk + embed → INSERT 新数据
    │             → UPSERT file_meta
    ├── 文件删除  → DELETE chunks / vec_chunks / file_meta
    └── 无变化    → 跳过，不做任何 IO 或推理
```

哈希使用 **SHA-256**（而非 mtime）作为变更判断依据，避免 mtime 在 git checkout、文件复制等场景下误判。

### 端到端数据流示例

```
查询: "AudioPlayer initialization"
  │
  ▼
tokenizer.json → [101, 3452, 7654, ...] (token ids)
  │
  ▼
VESO-25M → last_hidden_state [1, 8, 768]
  │
  ▼
mean-pool → [0.031, -0.142, 0.089, ...] (768 维)
  │
  ▼
L2-norm  → 单位向量

  ↓  sqlite-vec KNN (k=5)

  commons/audioplayer/.../SpeechPlayerService.ets  score=0.783
  features/mine/.../UserNetFunc.ets                score=0.759
  ...
```

---

## Project layout

```
src/
  main.rs       CLI entry point; build (incremental) and search commands
  chunker.rs    Regex-based .ets boundary detector and chunk builder
  tokenizer.rs  HuggingFace tokenizer wrapper (BPE, pad/truncate to 512)
  embedder.rs   ONNX session wrapper; masked mean-pool + L2-norm
  store.rs      SQLite + sqlite-vec store; incremental update logic
  index.rs      Legacy flat vector store (kept for unit tests)
  perf.rs       Timers, memory snapshots, JSON/Markdown report generation
hmosworld-master/
  ...           HarmonyOS World sample app — used as test corpus
```

---

## Incremental update strategy

The `file_meta` table tracks `(path, sha256, mtime)` for every indexed file.
On each `build` run:

1. Walk the repo and hash every `.ets` file.
2. Diff against `file_meta`:
   - **New** → chunk + embed + insert into `chunks` / `vec_chunks`
   - **Modified** → delete old rows, re-chunk + embed + insert
   - **Deleted** → delete rows from all three tables
   - **Unchanged** → skip entirely
3. Upsert `file_meta` for new/modified files.

Use `-Force` to wipe the database and rebuild from scratch.

---

## Running the unit tests

```powershell
.\test.ps1                      # all tests
.\test.ps1 -Filter chunker      # tests whose name contains "chunker"
.\test.ps1 -Filter store        # tests for the SQLite store
.\test.ps1 -Verbose             # show println! output for passing tests
```

The test suite covers:

- `chunker` — boundary detection regex and chunk count on synthetic ArkTS
- `index`   — cosine ranking order and bincode round-trip (legacy)
- `store`   — insert/search/delete on an in-memory SQLite database

---

## Crate dependencies

| Crate | Purpose |
|---|---|
| `ort 2.0.0-rc.12` + `ort-sys` | ONNX Runtime bindings; DLL loaded via `LoadLibraryW` + `ort::set_api` to bypass Windows Smart App Control hang on `LoadLibraryExW` |
| `tokenizers 0.21` | HuggingFace tokenizer (BPE / `onig` regex backend) |
| `rusqlite 0.32` | SQLite bindings (bundled SQLite, no system dependency) |
| `sqlite-vec 0.1.8-alpha.1` | sqlite-vec extension — `vec0` virtual table for vector search |
| `sha2 0.10` | SHA-256 content hashing for change detection |
| `clap 4` | CLI argument parsing (`derive` API) |
| `serde` / `serde_json` | JSON serialisation of reports |
| `bincode 1` | Binary serialisation (legacy index) |
| `anyhow` | Ergonomic error handling |
| `walkdir` | Recursive directory traversal |
| `windows 0.58` | Windows RSS / peak-working-set memory stats |
