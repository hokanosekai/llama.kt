# llama.kt

Kotlin/Android JNI binding for [llama.cpp](https://github.com/ggml-org/llama.cpp). Runs GGUF models on-device with GPU acceleration via OpenCL. Extracted and adapted from [llama.rn](https://github.com/a16z/llama.rn).

Designed to be consumed as a standalone Android library by the tensai app — no server, no cloud, inference runs on the device.

## Status

**WIP / scaffolding** — module structure and build configuration in place, JNI bindings and CMake integration coming next.

## License

Apache 2.0 — see [LICENSE](LICENSE). Third-party dependencies listed in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
