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
| `onnxruntime.dll` (Windows) | Bundled with **Lingma** at `%LOCALAPPDATA%\.lingma\env\onnxruntime.dll`, or downloadable from the [ONNX Runtime releases page](https://github.com/microsoft/onnxruntime/releases) |
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
.\build.ps1
```

Default parameters (override with flags):

| Parameter | Default | Description |
|---|---|---|
| `-Repo` | `hmosworld-master` | Root directory to scan for `.ets` files |
| `-Model` | DevEco Studio VESO-25M path | Directory containing `tokenizer.json` and `model_fp16.onnx` |
| `-Ort` | `%LOCALAPPDATA%\.lingma\env\onnxruntime.dll` | Path to `onnxruntime.dll` |
| `-Out` | `index.bin` | Output binary index file |
| `-Batch` | `16` | ONNX inference batch size (16–32 recommended) |
| `-Report` | `report` | Stem for report files: writes `<stem>.json` and `<stem>.md` |
| `-SkipCompile` | off | Skip `cargo build --release` and use the existing binary |

Example with custom paths:

```powershell
.\build.ps1 -Repo "path\to\my\repo" -Batch 32 -Report my_report
```

> **Tip:** set `RUST_LOG` to control log verbosity.
> `RUST_LOG=info` (the default) prints progress; `RUST_LOG=warn` suppresses it.

### Console output

```
[INFO] Chunking .ets files under "hmosworld-master" ...
[INFO]   42 files → 318 chunks  (3.21ms)
[INFO] Loading tokenizer and ONNX model ...
[INFO]   Model loaded (1.84s).
[INFO] Embedding 318 chunks (batch_size=16) ...
[INFO]   50/318 chunks embedded
[INFO]   100/318 chunks embedded
  ...
[INFO]   Embedding done (12.30s)
[INFO] Index saved to "index.bin"

====== Build Stats ======
  Repo              : hmosworld-master
  Files processed   : 42
  Chunks created    : 318
  Batch size        : 16
--------------------------
  Chunking time     : 3.21ms
  Model load time   : 1.84s
  Embedding time    : 12.30s
  Index build time  : 841µs
  Index save time   : 6.12ms
  TOTAL             : 14.15s
  Throughput        : 22.5 chunks/s
--------------------------
  Mem before (RSS)  : 14.3 MB
  Mem after  (RSS)  : 312.8 MB
  Mem delta         : 298.5 MB
  Peak working set  : 421.0 MB
--------------------------
  Index file size   : 0.94 MB
=========================
```

### Metrics JSON (`report.json`)

When `--report report` is passed, the following file is written automatically:

```json
{
  "generated_at_unix": 1742557200,
  "generated_at": "2026-03-21T10:00:00Z",
  "repo_path": "hmosworld-master",
  "model_path": "C:\\...\\VESO-25M",
  "batch_size": 16,
  "index_out": "index.bin",
  "total_files": 42,
  "total_chunks": 318,
  "timing_ms": {
    "chunking": 3,
    "model_load": 1840,
    "embedding": 12300,
    "index_build": 1,
    "index_save": 6,
    "total": 14150
  },
  "throughput_chunks_per_sec": 22.48,
  "memory_mb": {
    "before_rss": 14.3,
    "after_rss": 312.8,
    "delta": 298.5,
    "peak_working_set": 421.0
  },
  "index_size_mb": 0.94
}
```

A companion `report.md` Markdown table is also written alongside the JSON.

---

## Step 2 — Query the index

```powershell
.\search.ps1 -Query "AudioPlayer initialization"
.\search.ps1 -Query "network request" -Top 10
```

| Parameter | Default | Description |
|---|---|---|
| `-Query` | **required** | Natural-language or code query |
| `-Index` | `index.bin` | Index file produced by `build.ps1` |
| `-Model` | DevEco Studio VESO-25M path | Model directory (tokenizer) |
| `-Ort` | Lingma `onnxruntime.dll` | Path to `onnxruntime.dll` |
| `-Top` | `5` | Number of results to return |

### Example output

```
=== Top 5 results for: "AudioPlayer initialization" ===

[1] score=0.9312  features/audio/src/main/ets/components/AudioPlayer.ets  L1–48
    | @Component
    | export struct AudioPlayer {
    |   private player: media.AVPlayer | null = null

[2] score=0.8947  features/audio/src/main/ets/viewmodel/AudioViewModel.ets  L12–35
    | initPlayer() {
    |   this.avPlayer = media.createAVPlayer()
    |   this.avPlayer.on('stateChange', ...)
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
| `ort 2.0.0-rc.12` | ONNX Runtime bindings (`load-dynamic` for DLL loading) |
| `tokenizers 0.21` | HuggingFace tokenizer (BPE / `onig` regex backend) |
| `clap 4` | CLI argument parsing (`derive` API) |
| `serde` / `serde_json` | JSON serialisation of reports |
| `bincode 1` | Binary serialisation of the index |
| `rayon` | Parallel iterator support |
| `anyhow` | Ergonomic error handling |
| `walkdir` | Recursive directory traversal |
| `windows 0.58` | Windows RSS / peak-working-set memory stats |
