# How to run

```
bun install
bun run tauri dev
```

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
