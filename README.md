<p align="center">
  <a href="https://deepwiki.com/tleyden/tauri2-local-llm"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
</p>

This is part of a series of prototyping repos:

1. [Local model speech-to-text transcription library running from a Tauri2 desktop app](https://github.com/tleyden/tauri2-stt)
2. [Text-to-speech local model from Tauri/rust](https://github.com/tleyden/tauri2-qwen3-tts)
3. (this repo) [Gemma4-12b from Tauri/rust](https://github.com/tleyden/tauri2-local-llm)

These were created as part of prototyping the different options for use in a few apps I'm building: [Fluensy](https://fluensy.app) (foreign language learning app for professionals) and [brain3](https://github.com/tleyden/brain3) (MCP server for markdown vaults)

## P0 Requirements

1. Supports gemma4 12b
2. Works with the native audio feature of gemma4
3. Runs on macOS

## P1 Requirements

1. Other LLM models 
2. Linux/Windows

## Supported approaches

Only one for now, which has been working well so far. In the future I plan to compare other approaches.

1. [llama.cpp native integration via FFI (llama-cpp-2)](llama-cpp-ffi/README.md)

## Audio benchmark

Metric: generated model tokens divided by end-to-end audio prompt wall time,
excluding model load. Current test audio is
`llama-cpp-ffi/testdata/test_tts.wav`.

| Approach | Model | Hardware | Generated tokens | Elapsed seconds | Tokens/sec |
| --- | --- | --- | ---: | ---: | ---: |
| [llama.cpp native integration via FFI (llama-cpp-2)](llama-cpp-ffi/README.md) | Gemma 4 12B IT Q5_K_S | M2 MacBook Pro Max, 64GB unified memory, Metal | 96 | 5.82 | 16.48 |

## Design notes - best integration strategy?

### Option 1: crabnebula-dev/tauri-plugin-llm

#### Risks

1. Does it support gemma4?  Open Issue: https://github.com/crabnebula-dev/tauri-plugin-llm/issues/22

### Option 2: llama.cpp sidecar process

#### Risks

1. Avoiding orphaned processes

### Option 3: llama.cpp native integration via FFI - Implemented

Status: implemented in [`llama-cpp-ffi`](llama-cpp-ffi/README.md).

#### Strengths

1. Gemma4 + native audio support.  Issue that shows it's supported: https://github.com/utilityai/llama-cpp-rs/pull/1000

#### Risks

1. Can crash the Tauri2 process
2. More complex build toolchain

### Option 4: mlx-swift-lm with swift toolchain

#### Risks

1. More complicated toolchain
2. Native audio support not yet supported: https://github.com/ml-explore/mlx-swift-lm/issues/393

### Option 5: mlx-rs 

#### Risks

1. Gemma4 support pending: https://github.com/oxiglade/mlx-rs/pull/356

## Conclusion

I ended up going with Option 3: llama.cpp native integration via FFI, and it's working great so far.  See [`llama-cpp-ffi`](llama-cpp-ffi/README.md)
