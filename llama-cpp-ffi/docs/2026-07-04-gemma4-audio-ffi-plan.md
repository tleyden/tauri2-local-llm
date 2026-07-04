# Tauri 2 + Gemma 4 native audio via llama.cpp FFI (llama-cpp-2)

## Context

`tauri2-local-llm` is a prototyping repo comparing strategies for running local
models (Gemma 4) inside a Tauri 2 desktop app on macOS. The README lists five
options; this plan implements **Option 3: llama.cpp native integration via
FFI**, using the `utilityai/llama-cpp-rs` crate (`llama-cpp-2`), per the user's
explicit choice. Goal for this milestone: prove the riskiest P0 requirement —
Gemma 4's **native audio** input — works via the FFI, not just a bare CLI.

**Mandate (per repo `AGENTS.MD`): this repo is research/validation only.**
No Tauri commands, no frontend wiring, no shipped app — just unit/e2e tests
against a shared `core` module. The Tauri scaffold stays in place only
because the validated logic will later be dropped into a real Tauri 2 app;
that integration is out of scope here.

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
src-tauri (Rust crate, lib target only — no commands/frontend wired up)
  src/core/
    ├─ engine.rs   Engine { backend: LlamaBackend, model: LlamaModel, mtmd: MtmdContext, ctx, sampler }
    │              Engine::load(model_path, mmproj_path) -> Result<Self>
    │              Engine::prompt_text(&mut self, prompt: &str) -> Result<String>
    │              Engine::prompt_audio(&mut self, wav_path: &Path, prompt: &str) -> Result<String>
    └─ mod.rs      re-exports; path helpers for locating .models/... GGUFs
  tests/
    ├─ hello_world_test.rs        (Test 1)
    └─ audio_transcription_test.rs (Test 2)
```

`Engine` is the one piece of shared logic, factored out of the tests so each
test file is just: load, call, assert. No `tauri::State`, no async
command handlers, no sampling-loop duplication between call sites — that's
the only "refactor" this milestone needs.

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

4. **Build the `core` module** (`src-tauri/src/core/engine.rs`): factor the
   backend/model/mtmd-context load and the tokenize → `eval_chunks()` →
   greedy-sampling-loop response logic out of any single test, mirroring
   `examples/mtmd/src/mtmd.rs`'s calling convention rather than reinventing
   it. `Engine::load` takes explicit model/mmproj paths (tests resolve these
   from `.models/gemma-4-12b-it-Q5_K_S/`); `prompt_text` skips mtmd/bitmap
   entirely, `prompt_audio` uses `MtmdBitmap::from_file()` + the chat-template
   marker pattern. Add `pub mod core;` to `src-tauri/src/lib.rs` so integration
   tests under `tests/` can `use llama_cpp_ffi_lib::core::Engine`.

5. **Write the two tests** described in the Test Plan below, under
   `src-tauri/tests/`.

## Test Plan

Progressively harder unit/e2e tests, run via `cargo test` from `src-tauri/`.
Both need the real model files present locally (`.models/gemma-4-12b-it-Q5_K_S/`,
gitignored) — no mocking the FFI, that would defeat the point of this
validation. Mark them `#[ignore]`-free but expect them to be slow (multi-GB
model load); that's acceptable for this prototype per `AGENTS.MD`.

1. **`hello_world_test.rs` — model loads, text round-trip.**
   - `Engine::load(model_path, mmproj_path)` succeeds (no panic/error).
   - `engine.prompt_text("Say hello.")` (or similar) returns a non-empty
     `String`.
   - This is the cheapest possible smoke test: it validates the toolchain
     (CMake/Xcode CLT build), the GGUF loads, and basic text generation works
     — before audio enters the picture at all.

2. **`audio_transcription_test.rs` — native audio round-trip.**
   - Same `Engine`, but call `engine.prompt_audio("testdata/test_tts.wav",
     "Transcribe this audio exactly.")`.
   - Assert `MtmdContext::support_audio()` is `true` and
     `get_audio_sample_rate()` reports `16000` before running the prompt
     (fail fast with a clear message if the crate/model combo doesn't
     support audio).
   - Assert the response contains "hello world" (case-insensitive
     substring match — a fixed transcript, not an exact-match generation).

**Backlog (not implemented this milestone — future harder tests once 1 & 2
pass):**
- Combined audio + question prompt (e.g. asking about audio content, not
  just transcribing it).
- Oversized/out-of-spec audio input (>30s, non-16kHz) — confirm graceful
  error vs. crash.
- Two sequential `Engine` calls confirming no state leaks between prompts.
- Multiple `Engine::load()` calls to approximate the "load once, reuse"
  behavior a future Tauri `State` would need — this repo won't build that
  state wrapper, but the test can prove the underlying assumption holds.

## Verification

- `cargo build --features metal` compiles cleanly on macOS with no manual
  llama.cpp setup beyond Xcode CLT + CMake.
- `cargo test --features metal` runs both tests in the Test Plan above and
  they pass against the real local model + `testdata/test_tts.wav`.

## Known risks (carried forward, not solved here)

- FFI crash = whole test process crash (inherent to Option 3, per README) —
  same risk would apply to a future Tauri app, just observed here as a
  failed `cargo test` run instead.
- 12B GGUF + mmproj is several GB; each test run pays full model-load time —
  acceptable for a prototype, no shared-fixture/singleton optimization here.
- No in-app model download/verification flow — out of scope, this repo has
  no app.
