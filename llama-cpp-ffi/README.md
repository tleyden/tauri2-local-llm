# How to run

```
bun install
bun run tauri:dev
```

On macOS this prototype always builds `llama-cpp-2` with Metal enabled. The
`src-tauri` crate enables its `metal` and `mtmd` features by default, and the
Tauri scripts pass `--features metal` explicitly:

```
bun run tauri:dev
bun run tauri:build
```

For Rust-only validation from `src-tauri/`, use:

```
cargo build --features metal
cargo test --features metal
```

# Benchmarks

The Rust integration tests print simple local inference timings. Run them with
`--show-output` so Cargo keeps the benchmark line visible after a passing test.

From `src-tauri/`, run the audio transcription benchmark:

```
cargo test --features metal --test audio_transcription_test \
  gemma_4_native_audio_returns_expected_process_management_answer -- --show-output
```

By default, the benchmark uses Q8_0 for the K/V cache. To compare it with the
old default F16 K/V cache, set `LLAMA_CPP_FFI_KV_CACHE`:

```
LLAMA_CPP_FFI_KV_CACHE=q8_0 cargo test --features metal --test audio_transcription_test \
  gemma_4_native_audio_returns_expected_process_management_answer -- --show-output

LLAMA_CPP_FFI_KV_CACHE=f16 cargo test --features metal --test audio_transcription_test \
  gemma_4_native_audio_returns_expected_process_management_answer -- --show-output
```

The output includes a line like:

```
audio benchmark: input_tokens=503 input_positions=503 generated_tokens=96 prefill_seconds=1.99 decode_seconds=4.63 decode_tokens_per_second=20.74 total_seconds=6.62 total_tokens_per_second=14.51
```

For a text-only sanity benchmark:

```
cargo test --features metal --test hello_world_test \
  gemma_4_model_loads_and_answers_text_prompt -- --show-output
```

Compare `decode_tokens_per_second` for decode-only speed and
`total_tokens_per_second` for the full prompt path.

# Local models

This prototype expects local GGUF model files outside git. Keep them mounted
under `.models/` or `models/`; both directories are ignored.

To reuse a Jan-downloaded Gemma 4 model without copying the multi-GB files:

```
mkdir -p .models
ln -s "$HOME/Library/Application Support/Jan/data/llamacpp/models/gemma-4-12b-it-Q5_K_S" \
  .models/gemma-4-12b-it-Q5_K_S
```

The resulting local paths are:

```
.models/gemma-4-12b-it-Q5_K_S/model.gguf
.models/gemma-4-12b-it-Q5_K_S/mmproj.gguf
```
