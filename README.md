# llama.kt

Kotlin/Android JNI binding for [llama.cpp](https://github.com/ggml-org/llama.cpp). Loads any GGUF model and runs inference on-device — no server, no cloud. The native C++ engine is extracted from [llama.rn](https://github.com/mybigday/llama.rn) (React Native binding), stripped of the RN layer and re-exposed through a plain Kotlin API.

Built as the inference layer for the [tensai](https://tens.ai) app, but usable standalone.

## Status

Working. Validated on-device (OPPO / MediaTek Dimensity 900, arm64): loads a GGUF, applies the model's chat template, streams tokens, stops on EOS. ~20 tok/s CPU on Qwen2.5-0.5B (q4_K_M).

| Backend | Status |
|---|---|
| CPU (arm64, `armv8.2-a+dotprod`) | ✅ works everywhere |
| OpenCL (Adreno 700+, Snapdragon) | ✅ compiled in — untested on Adreno hardware yet |
| Vulkan (Mali, Xclipse, …) | ⏳ planned — needed for non-Adreno GPUs |
| CPU per-feature dispatch (i8mm/SVE on capable cores) | ⏳ planned — currently a single conservative variant |

Multimodal (vision via `mtmd`) is vendored but not yet wired through the Kotlin API.

## Requirements

- Android NDK r27+, `arm64-v8a` (only ABI targeted)
- `minSdk 29` (Android 10 — the whole point is on-device; older ABIs untargeted)
- JDK 21, Android SDK platform 35

## Install (git submodule)

```bash
git submodule add https://github.com/hokanosekai/llama.kt libs/llama.kt
git -C libs/llama.kt submodule update --init --recursive
bash libs/llama.kt/scripts/bootstrap.sh        # vendors llama.cpp sources (required before first build)
```

`settings.gradle.kts`:
```kotlin
include(":llama-kt")
project(":llama-kt").projectDir = file("libs/llama.kt/llama-kt")
```

`build.gradle.kts`:
```kotlin
dependencies { implementation(project(":llama-kt")) }
```

(Maven Central publishing may come later; submodule is the supported path for now.)

## Quick start

```kotlin
import com.tensai.llamakt.*

val engine = LlamaEngine()
engine.load(path = "/path/to/model.gguf", nGpuLayers = 0, nCtx = 4096)

val messages = listOf(ChatMessage("user", "Who are you?"))
engine.chat(messages).collect { token -> print(token) }   // streams, stops on EOS

engine.free()
```

`chat()` applies the model's built-in chat template (read from the GGUF), so it works with any instruct model without hardcoding a format.

## API

```kotlin
class LlamaEngine {
    fun load(path: String, nGpuLayers: Int = 0, nCtx: Int = 4096)
    fun tokenizeCount(text: String): Int
    fun kvUsed(): Int                 // KV cache cells used = context budget
    fun interrupt()                   // cancels an in-flight generation (incl. prefill)
    fun saveSession(path: String): Int
    fun loadSession(path: String): Int
    fun free()
}

data class ChatMessage(val role: String, val content: String)
fun interface TokenCallback { fun onToken(token: String) }

// extensions (Flow)
fun LlamaEngine.chat(messages: List<ChatMessage>, nPredict: Int = 512): Flow<String>
fun LlamaEngine.decode(prompt: String, nPredict: Int = 512): Flow<String>   // raw, no template
fun LlamaEngine.formatChat(messages: List<ChatMessage>): String
```

Generation stops on the model's EOS token or after `nPredict` tokens (hard cap against runaway generation). Flows run on `Dispatchers.Default`; collecting on the main thread is safe.

Context budget: `kvUsed()` returns cells used, compare against your `nCtx` to show a fill bar. Persist a conversation's KV with `saveSession`/`loadSession` (restore requires the same model + context params).

## Example

[`example/`](example/) is a minimal Compose app (pick a GGUF, chat, live tok/s, stop) — the reference integration.

## Building from source

`scripts/bootstrap.sh` copies llama.cpp sources from the pinned submodule into `llama-kt/src/main/cpp/` and rewrites symbols (`ggml_` → `lm_ggml_`) to avoid collisions. `scripts/build-opencl.sh` builds the `libOpenCL.so` ICD stub from the Khronos submodules. Both run against the `third_party/` submodules — init them first.

```bash
git submodule update --init --recursive
bash scripts/bootstrap.sh
bash scripts/build-opencl.sh
./gradlew :example:assembleDebug
```

## Upstream tracking

The native engine mirrors llama.rn's `cpp/rn-llama.*`, `rn-completion.*`, `rn-mtmd.hpp` + the `ggml-opencl/` backend. We extract files, not fork the repo (avoids merge conflicts from the JS side). To pull upstream changes: bump the `third_party/llama.cpp` submodule deliberately (breaking API changes are frequent), re-run `bootstrap.sh`, port relevant `rn-*` changes from a fresh llama.rn checkout, rebuild. The `cpp/jsi/` React Native adapter is intentionally dropped.

## License

Apache 2.0 — see [LICENSE](LICENSE). Third-party notices in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md): llama.rn (MIT), llama.cpp (MIT), OpenCL-ICD-Loader (Apache 2.0).
