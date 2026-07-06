# Roadmap

What's planned for llama.kt, in rough priority order. This file tracks direction, not commitments — items move between sections as constraints become clear.

## v1.1 — Distribution & dispatch

- **Published AAR** (Maven Central) — install with `implementation("io.github.hokanosekai:llama-kt:…")` instead of a git submodule. The submodule path stays supported; this is about lowering the barrier to consume the lib.
- **CPU per-feature dispatch** — ship multiple CPU variants (baseline / dotprod / i8mm / SVE) and select at runtime, to keep i8mm speed on capable cores without crashing older ones.
- **Runtime backend auto-selection** — pick Vulkan / OpenCL / CPU per device automatically, with a manual override.
- **Multimodal** — wire the vendored `mtmd` (vision) through the Kotlin API.

## v1.x — Hardening & coverage

- **Exact KV cache sizing** — expose the GGUF keys needed to compute KV memory precisely (`n_head_kv`, `n_embd_head`) in `GgufMetadata`, so apps can predict memory before loading.
- **Interrupt during prefill** — validate that `interrupt()` cancels promptly while a long prompt is still being processed, not just during generation.
- **OpenCL on Adreno** — the backend is compiled in but untested on Adreno hardware; validate and add perf numbers.
- **Benchmarks** — KleidiAI on/off impact, mmap vs no-mmap+mlock load strategies.

## Upstream

- **Drop the Vulkan UMA patch** once [ggml-org/llama.cpp#23057](https://github.com/ggml-org/llama.cpp/issues/23057) lands upstream — the vendored `0001-vulkan-uma-descriptor-ceildiv.patch` becomes unnecessary.

## Candidates (not committed)

Ideas under consideration — no timeline, may never happen:

- **LoRA adapters at runtime** — load/unload small adapter weights on top of a base GGUF (`loadAdapter(path, scale)`), specialize a model without shipping a new one.
- **Embeddings API** — expose text → vector embedding, the building block for on-device semantic search and RAG. Supported by llama.cpp, not yet surfaced in the Kotlin API.
- **Grammar / JSON-schema constrained output** — constrain sampling with a GBNF grammar or JSON schema so structured output is guaranteed parseable. Main enabler for reliable tool calling.

## Non-goals

Out of scope for this lib, by design:

- **HTTP/API server** — llama.kt is an inference binding. Serving belongs to the consuming app.
- **Model download / management** — the lib takes a file path; fetching, storing and versioning models is app territory.
- **Agent loop / tool orchestration** — prompt→model→tool-call loops live above the engine layer.
