This is a prototyping repo to compare the various ways to use local models like Gemma4 in a Tauri2 desktop app, since I am planning to integrate this into the language learning app I'm building: [Fluensy](https://fluensy.app)

Right now I mainly need it on macOS, so there is a heavy bias towards that platform.  But ideally I want this to work on all platforms.

## P0 Requirements

1. Supports gemma4 12b
2. Works with the native audio feature of gemma4
3. Runs on macOS

## P1 Requirements

1. Other LLM models 
2. Linux/Windows

## Supported approaches

1. llama.cpp native integration via FFI (llama-cpp-2)


## Design notes - best integration strategy?

### Option 1: crabnebula-dev/tauri-plugin-llm

#### Risks

1. Does it support gemma4?  Open Issue: https://github.com/crabnebula-dev/tauri-plugin-llm/issues/22

### Option 2: llama.cpp sidecar process

#### Risks

1. Avoiding orphaned processes

### Option 3: llama.cpp native integration via FFI

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
