# RustEmbedding

Semantic code search for **ArkTS / HarmonyOS `.ets` files**.  
The tool chunks source files by structural boundaries, embeds each chunk with a
local ONNX model (VESO-25M, FP16), stores the vectors in a flat binary index,
and answers natural-language or code queries with cosine-similarity retrieval.

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
| `build.ps1` | Compile the project, index a repo, write `report.json` / `report.md` |
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
  -Out    "index.bin" `
  -Batch  16 `
  -Report "report"
```

Default parameters (override with flags):

| Parameter | Default | Description |
|---|---|---|
| `-Repo` | `hmosworld-master` | Root directory to scan for `.ets` files |
| `-Model` | DevEco Studio VESO-25M path | Directory containing `tokenizer.json` and `model_fp16.onnx` |
| `-Ort` | `C:\Users\<you>\Downloads\onnxruntime-win-x64-1.24.4\lib\onnxruntime.dll` | Path to `onnxruntime.dll` (must be v1.24.x) |
| `-Out` | `index.bin` | Output binary index file |
| `-Batch` | `16` | ONNX inference batch size (16–32 recommended) |
| `-Report` | `report` | Stem for report files: writes `<stem>.json` and `<stem>.md` |
| `-SkipCompile` | off | Skip `cargo build --release` and use the existing binary |

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

### Console output

```
[INFO] Chunking .ets files under "hmosworld-master" ...
[INFO]   161 files → 788 chunks  (19ms)
[INFO] Loading tokenizer and ONNX model ...
[INFO]   Model loaded (0.29s).
[INFO] Embedding 788 chunks (batch_size=16) ...
[INFO]   64/788 chunks embedded
[INFO]   112/788 chunks embedded
  ...
[INFO]   Embedding done (23.17s)
[INFO] Index saved to "index.bin"

====== Build Stats ======
  Repo              : hmosworld-master
  Files processed   : 161
  Chunks created    : 788
  Batch size        : 16
--------------------------
  Chunking time     : 19ms
  Model load time   : 292ms
  Embedding time    : 23172ms
  Index build time  : 0ms
  Index save time   : 2ms
  TOTAL             : 23487ms
  Throughput        : 34.0 chunks/s
--------------------------
  Mem before (RSS)  : 5.1 MB
  Mem after  (RSS)  : 182.9 MB
  Mem delta         : 177.7 MB
  Peak working set  : 418.3 MB
--------------------------
  Index file size   : 2.95 MB
=========================
```

### Metrics JSON (`report.json`)

When `--report report` is passed, the following file is written automatically:

```json
{
  "generated_at_unix": 1742601722,
  "generated_at": "2026-03-22T04:02:02Z",
  "repo_path": "hmosworld-master",
  "model_path": "C:\\...\\VESO-25M",
  "batch_size": 16,
  "index_out": "index.bin",
  "total_files": 161,
  "total_chunks": 788,
  "timing_ms": {
    "chunking": 19,
    "model_load": 292,
    "embedding": 23172,
    "index_build": 0,
    "index_save": 2,
    "total": 23487
  },
  "throughput_chunks_per_sec": 34.0,
  "memory_mb": {
    "before_rss": 5.1,
    "after_rss": 182.9,
    "delta": 177.7,
    "peak_working_set": 418.3
  },
  "index_size_mb": 2.95
}
```

A companion `report.md` Markdown table is also written alongside the JSON.

---

## Step 2 — Query the index

```powershell
# Short form (uses index.bin + defaults)
.\search.ps1 -Query "AudioPlayer initialization"
.\search.ps1 -Query "network request" -Top 10

# Explicit paths
.\search.ps1 `
  -Query  "AudioPlayer initialization" `
  -Index  "index.bin" `
  -Model  "C:\Program Files\Huawei\DevEco Studio\plugins\codegenie-plugin\embedding_model\VESO-model\VESO-25M" `
  -Ort    "C:\Users\qinzh\Downloads\onnxruntime-win-x64-1.24.4\lib\onnxruntime.dll" `
  -Top    5
```

| Parameter | Default | Description |
|---|---|---|
| `-Query` | **required** | Natural-language or code query |
| `-Index` | `index.bin` | Index file produced by `build.ps1` |
| `-Model` | DevEco Studio VESO-25M path | Model directory (tokenizer) |
| `-Ort` | `C:\Users\<you>\Downloads\onnxruntime-win-x64-1.24.4\lib\onnxruntime.dll` | Path to `onnxruntime.dll` (must be v1.24.x) |
| `-Top` | `5` | Number of results to return |

### Example output

```
=== Top 5 results for: "AudioPlayer initialization" ===

[1] score=0.7834  commons\audioplayer\src\main\ets\service\SpeechPlayerService.ets  L72–72
    |         let filePath = path + TEMP_AUDIO_FILE_NAME;

[2] score=0.7586  features\mine\src\main\ets\service\UserNetFunc.ets  L166–166
    |         const aaid: string = await AAID.getAAID();

[3] score=0.7447  features\mine\src\main\ets\service\UserNetFunc.ets  L167–168
    |         const pushToken: string = await pushService.getToken();
    |         Logger.info(TAG, 'Get AAID successfully: %{public}s', aaid);
...
```

---

## Project layout

```
src/
  main.rs       CLI entry point; build and search commands
  chunker.rs    Regex-based .ets boundary detector and chunk builder
  tokenizer.rs  HuggingFace tokenizer wrapper (BPE, pad/truncate to 512)
  embedder.rs   ONNX session wrapper; masked mean-pool + L2-norm
  index.rs      Flat vector store (cosine similarity, bincode serialisation)
  perf.rs       Timers, memory snapshots, JSON/Markdown report generation
hmosworld-master/
  ...           HarmonyOS World sample app — used as test corpus
```

---

## Running the unit tests

```powershell
.\test.ps1                      # all tests
.\test.ps1 -Filter chunker      # tests whose name contains "chunker"
.\test.ps1 -Verbose             # show println! output for passing tests
```

The test suite covers:

- `chunker` — boundary detection regex and chunk count on synthetic ArkTS
- `index`   — cosine ranking order and bincode round-trip

---

## Crate dependencies

| Crate | Purpose |
|---|---|
| `ort 2.0.0-rc.12` + `ort-sys` | ONNX Runtime bindings; DLL loaded via `LoadLibraryW` + `ort::set_api` to bypass Windows Smart App Control hang on `LoadLibraryExW` |
| `tokenizers 0.21` | HuggingFace tokenizer (BPE / `onig` regex backend) |
| `clap 4` | CLI argument parsing (`derive` API) |
| `serde` / `serde_json` | JSON serialisation of reports |
| `bincode 1` | Binary serialisation of the index |
| `rayon` | Parallel iterator support |
| `anyhow` | Ergonomic error handling |
| `walkdir` | Recursive directory traversal |
| `windows 0.58` | Windows RSS / peak-working-set memory stats |
