# Tauri 2 + Gemma 4 native audio via llama.cpp FFI (llama-cpp-2)

## Context

`tauri2-local-llm` is a prototyping repo comparing strategies for running local
models (Gemma 4) inside a Tauri 2 desktop app on macOS. The README lists five
options; this plan implements **Option 3: llama.cpp native integration via
FFI**, using the `utilityai/llama-cpp-rs` crate (`llama-cpp-2`), per the user's
explicit choice. Goal for this milestone: prove the riskiest P0 requirement —
Gemma 4's **native audio** input — works end-to-end inside a real Tauri 2 app
on macOS, not just a bare CLI.

Research grounding (verified against the actual repo/crate, not assumed):
- `llama-cpp-2` added Gemma 4 audio support in PR
  [utilityai/llama-cpp-rs#1000](https://github.com/utilityai/llama-cpp-rs/pull/1000)
  (merged 2026-04-22), bumping the vendored llama.cpp to `b8783`.
- Upstream llama.cpp had a further fix, **PR #24118 "Fix Gemma 4 Unified
  conversion"** (merged 2026-06-04), specifically for Gemma 4 12B audio
  processing bugs. The llama-cpp-rs submodule was bumped past this on
  2026-06-08 (commit `e9885c6`). Crates.io `llama-cpp-2` **0.1.150**
  (published 2026-06-16) is the first published version that includes it.
  **Action: pin `llama-cpp-2 = "0.1.150"` or newer** — do not use 0.1.146 or
  earlier, they predate the 12B audio fix.
- The crate exposes multimodal via an `mtmd` Cargo feature wrapping
  libmtmd: `MtmdContext::init_from_file`, `MtmdBitmap::from_file` /
  `from_audio_data` (raw PCM f32), `MtmdContext::support_audio()`,
  `get_audio_sample_rate()`, `tokenize()`, `InputChunks::eval_chunks()`. A
  working reference CLI already exists in the upstream repo at
  `examples/mtmd/src/mtmd.rs` — our implementation should closely mirror it
  rather than reinvent the mtmd calling convention.
- Gemma 4 12B is the first mid-size Gemma with native audio; per its docs,
  safe audio input is **up to 30s, 16kHz mono**. Quantized GGUF + mmproj are
  published under `unsloth/gemma-4-12b-it-GGUF` and `ggml-org` HF repos.
- On macOS, `llama-cpp-2`'s own Cargo.toml auto-adds the `metal` feature to
  its sys crate for `aarch64`/`arm64`, but that's only for its *own* build —
  our downstream Cargo.toml still needs to request `features = ["mtmd",
  "metal"]` explicitly.
- Building requires a working C/C++ toolchain (Xcode Command Line Tools) and
  CMake, since `llama-cpp-sys-2`'s build script compiles vendored
  llama.cpp/ggml from source.

## Architecture

```
Tauri 2 app (macOS)
  src-tauri (Rust)
    ├─ tauri::State<Mutex<AudioEngine>>   (loaded once, lazily, on first use)
    │    ├─ LlamaBackend
    │    ├─ LlamaModel            (Gemma 4 12B GGUF)
    │    ├─ MtmdContext           (mmproj GGUF)
    │    └─ LlamaContext + sampler
    └─ #[tauri::command] run_audio_prompt(wav_path, prompt) -> Result<String, String>
  frontend (minimal HTML/JS or React from scaffold)
    └─ file picker (wav) + prompt textarea + button -> invoke("run_audio_prompt")
```

Inference runs via `tauri::async_runtime::spawn_blocking` so the ~10s+ model
load and per-request decode don't block the Tauri/webview event loop. This
does **not** fix the FFI blast-radius risk already flagged in the README (a
panic/abort inside llama.cpp still takes down the whole process) — that
tradeoff is accepted for this milestone, not solved.

## Steps

1. **Scaffold Tauri 2 project** in this repo using `npm create tauri-app@latest`
   (Rust backend, minimal frontend template — vanilla or React, doesn't
   matter for this prototype). Produces `src-tauri/`, `tauri.conf.json`,
   `package.json`. *(Already done — scaffold exists under `llama-cpp-ffi/`.)*

2. **Add `llama-cpp-2` dependency** in `src-tauri/Cargo.toml`:
   ```toml
   [target.'cfg(target_os = "macos")'.dependencies]
   llama-cpp-2 = { version = "0.1.150", features = ["mtmd", "metal"] }
   ```
   Confirm `cargo build` succeeds (this alone validates the CMake/Xcode CLT
   toolchain works) before writing any app logic.

3. **Acquire the model** manually for now (no in-app downloader in this
   milestone): fetch a Gemma 4 12B GGUF quant (e.g. Q4_K_M) and its mmproj
   file from `unsloth/gemma-4-12b-it-GGUF` or `ggml-org`'s HF repo, place
   under a gitignored `models/` dir.

4. **Standalone validation first**: before touching Tauri, adapt
   `examples/mtmd/src/mtmd.rs`'s logic into a throwaway `src-tauri/examples/`
   or `src-tauri/src/bin/` binary that loads the model + mmproj and runs one
   audio prompt from the CLI. This isolates "does Gemma 4 12B audio work with
   this crate version at all" from "does it work inside Tauri," so failures
   are easy to attribute. Confirm `MtmdContext::support_audio()` is `true` and
   `get_audio_sample_rate()` reports `16000`.

5. **Wire into Tauri**: port the validated logic into an `AudioEngine` struct
   in `src-tauri/src/audio_engine.rs` (load-once backend/model/mtmd-context,
   held in `tauri::State<Mutex<AudioEngine>>`, lazily initialized on first
   command invocation to keep app startup fast). Expose:
   ```rust
   #[tauri::command]
   async fn run_audio_prompt(state: State<'_, Mutex<AudioEngine>>, wav_path: String, prompt: String) -> Result<String, String>
   ```
   using `MtmdBitmap::from_file(wav_path, ...)`, the chat-template + marker
   pattern from the example, `tokenize()`, `eval_chunks()`, then a greedy
   sampling loop to produce the response string.

6. **Minimal frontend**: a file input (or hardcoded path for this prototype),
   a textarea for the instruction (e.g. "Transcribe this audio exactly."),
   a submit button calling `invoke('run_audio_prompt', {...})`, and a
   `<pre>` to show the result/errors.

## Verification

- `cargo build --features metal` (or via `cargo tauri build` config) compiles
  cleanly on macOS with no manual llama.cpp setup beyond Xcode CLT + CMake.
- Standalone binary (step 4) run against a real short WAV (≤30s, 16kHz mono
  per Gemma 4's documented safe range) produces a plausible transcript/answer
  printed to the terminal.
- `cargo tauri dev` launches the app; submitting the same WAV + prompt through
  the UI returns the same kind of output in the UI, and terminal logs show
  no panics.
- Confirm the model is loaded only once across multiple submissions (check
  logs/timing — second request should skip model-load time).

## Known risks (carried forward, not solved here)

- FFI crash = whole app crash (inherent to Option 3, per README).
- 12B GGUF + mmproj is several GB; first load will be slow — acceptable for a
  prototype, not addressed with progress UI/streaming in this milestone.
- No in-app model download/verification flow yet (manual placement only).
