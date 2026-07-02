# llama.kt

Kotlin/Android JNI binding for [llama.cpp](https://github.com/ggml-org/llama.cpp). Loads any GGUF model and runs inference on-device — no server, no cloud. The native C++ engine is extracted from [llama.rn](https://github.com/mybigday/llama.rn) (React Native binding), stripped of the RN layer and re-exposed through a plain Kotlin API.

Built as the inference layer for the [tensai](https://tens.ai) app, but usable standalone.

## Status

Working. Validated on-device (OPPO / MediaTek Dimensity 900, arm64): loads a GGUF, applies the model's chat template, streams tokens, stops on EOS. ~20 tok/s CPU on Qwen2.5-0.5B (q4_K_M).

| Backend | Status |
|---|---|
| CPU (arm64, `armv8.2-a+dotprod`) | ✅ works everywhere |
| OpenCL (Adreno 700+, Snapdragon) | ✅ compiled in — untested on Adreno hardware yet |
| Vulkan (Mali, Xclipse, any Vulkan 1.2+ GPU) | ✅ works — validated on Mali-G68 (Dimensity 900), includes a local UMA fix (see below) |
| CPU per-feature dispatch (i8mm/SVE on capable cores) | ⏳ planned — currently a single conservative variant |

Multimodal (vision via `mtmd`) is vendored but not yet wired through the Kotlin API.

**Vulkan UMA fix.** Mobile drivers (Mali, Adreno, CIX) report `maxComputeWorkGroupCount = UINT32_MAX`, which overflows a 32-bit `CEIL_DIV` in `ggml_vk_matmul` and under-requests descriptor sets — batched matmuls (Gemma 3n and others) then abort at model load ([upstream #23057](https://github.com/ggml-org/llama.cpp/issues/23057)). The vendored `ggml-vulkan.cpp` in this repo carries a 2-line 64-bit promotion fix, validated on Mali-G68. **Warning: re-running `bootstrap.sh` regenerates the vendored sources and drops this fix** until it lands upstream or bootstrap gains a patch step.

**CPU threads.** `load()` defaults to pinning inference threads to the big cores (detected via `cpuinfo_max_freq`). On big.LITTLE SoCs this measured up to 6× faster than llama.cpp's auto-detect, which lets efficiency cores drag the pool down.

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
    // nThreads: 0 = pin to big cores (default), -1 = llama.cpp auto, >0 = explicit
    fun load(path: String, nGpuLayers: Int = 0, nCtx: Int = 4096, nThreads: Int = 0)
    fun completion(prompt: String, params: SamplingParams = SamplingParams(), callback: TokenCallback)
    fun formatChat(messages: List<ChatMessage>): String
    fun tokenize(text: String): IntArray
    fun kvUsed(): Int                 // KV cache cells used = context budget
    fun interrupt()                   // cancels an in-flight generation (incl. prefill)
    fun saveSession(path: String): Int
    fun loadSession(path: String): Int
    fun availableBackends(): List<BackendInfo>   // static device enumeration
    fun activeBackend(): String                  // backend actually used by the loaded model
    fun free()

    companion object {
        fun detectBigCoreCount(): Int  // cores with cpuinfo_max_freq >= 85% of the fastest
    }
}

data class SamplingParams(
    val nPredict: Int = 512,       // hard cap on generated tokens
    val temperature: Float = 0.8f, // 0 = greedy/deterministic
    val topK: Int = 40,            // 0 = disabled
    val topP: Float = 0.95f,       // 1.0 = disabled
    val minP: Float = 0.05f,       // 0 = disabled
    val stopSequences: List<String> = emptyList(),  // never emitted to the stream
)

data class ChatMessage(val role: String, val content: String)
data class BackendInfo(val name: String, val description: String, val type: String)
fun interface TokenCallback { fun onToken(token: String) }

// extensions (Flow)
fun LlamaEngine.chat(messages: List<ChatMessage>, params: SamplingParams = SamplingParams()): Flow<String>
fun LlamaEngine.decode(prompt: String, params: SamplingParams = SamplingParams()): Flow<String>   // raw, no template
```

Generation stops on the model's EOS token or after `SamplingParams.nPredict` tokens (hard cap against runaway generation). Sampling defaults match llama.cpp; `temperature = 0` gives reproducible greedy output. Flows run on `Dispatchers.Default`; collecting on the main thread is safe.

Native llama.cpp/ggml logs (including `GGML_ABORT` assert messages) are forwarded to logcat under the `llama.cpp` tag, so native crashes are diagnosable from `adb logcat`.

Context budget: `kvUsed()` returns cells used, compare against your `nCtx` to show a fill bar. Persist a conversation's KV with `saveSession`/`loadSession` (restore requires the same model + context params).

## Example

[`example/`](example/) is a Compose bench app — the reference integration. Pick a GGUF, choose backend (CPU/GPU) and thread count, tune sampling (max tokens, temperature, top-k, top-p), generate with live stats (tok/s, KV, RAM, CPU, collapsible graphs). Preset prompts include short-output "sampling test" chips to observe determinism/variance across runs.

It is also drivable headless from adb for benches and crash repros:

```bash
# put a model in the app cache (SAF picking is not adb-scriptable)
adb shell 'cat /sdcard/Download/model.gguf | run-as com.tensai.llamakt.example sh -c "cat > cache/model.gguf"'

# load + generate automatically, results logged as "LlamaKtBench: bench: ..."
adb shell am start -n com.tensai.llamakt.example/.MainActivity \
  --ez autorun true --ez gpu true --es prompt "Hello"
```

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

**Local patches on vendored code** (lost if you re-run `bootstrap.sh` — re-apply manually until bootstrap gains a patch step):
- `ggml-vulkan/ggml-vulkan.cpp`: 64-bit `CEIL_DIV` promotion in `ggml_vk_matmul` descriptor set requests + diagnostic log before the pool assert (fixes [#23057](https://github.com/ggml-org/llama.cpp/issues/23057) on UMA GPUs).

## Roadmap

- **Published AAR** (Maven Central) — install with `implementation("io.github.hokanosekai:llama-kt:…")` instead of a git submodule. The submodule path stays supported; this is about lowering the barrier to consume the lib.
- **Runtime backend auto-selection** — pick Vulkan / OpenCL / CPU per device automatically, with a manual override.
- **CPU per-feature dispatch** — ship multiple CPU variants (baseline / dotprod / i8mm / SVE) and select at runtime, to keep i8mm speed on capable cores without crashing older ones.
- **Multimodal** — wire the vendored `mtmd` (vision) through the Kotlin API.
- **Patch step in bootstrap** — keep local fixes to vendored code (Vulkan UMA fix) as `patches/*.patch` applied automatically, so `bootstrap.sh` stops silently dropping them.

## License

Apache 2.0 — see [LICENSE](LICENSE). Third-party notices in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md): llama.rn (MIT), llama.cpp (MIT), OpenCL-ICD-Loader (Apache 2.0).
