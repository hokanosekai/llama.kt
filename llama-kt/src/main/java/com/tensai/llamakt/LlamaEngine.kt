package com.tensai.llamakt

import org.json.JSONArray
import org.json.JSONObject

/** A single message in a chat conversation. */
data class ChatMessage(val role: String, val content: String)

/**
 * Sampling parameters for a completion.
 *
 * [nPredict] — hard cap on the number of generated tokens.
 * [temperature] — softmax temperature. Lower = more deterministic,
 *   higher = more random. 0 disables randomness (greedy). Default 0.8.
 * [topK] — keep only the K most probable tokens before sampling.
 *   0 disables the filter. Default 40.
 * [topP] — nucleus sampling: keep the smallest set of tokens whose
 *   cumulative probability reaches P. 1.0 disables the filter. Default 0.95.
 * [minP] — drop tokens whose probability is below minP times the top
 *   token's probability. 0 disables the filter. Default 0.05.
 * [stopSequences] — strings that end generation when they appear in the
 *   output. The matched sequence is not emitted (partial matches are held
 *   back until resolved), so the stream stays clean.
 *
 * Defaults match llama.cpp's common_params_sampling.
 */
data class SamplingParams(
    val nPredict: Int = 512,
    val temperature: Float = 0.8f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val minP: Float = 0.05f,
    val stopSequences: List<String> = emptyList(),
)

/**
 * Information about a single ggml backend device.
 * [name] — device name as reported by ggml (e.g. "Mali-G68 MC4")
 * [description] — backend registry name (e.g. "Vulkan", "CPU")
 * [type] — "cpu", "gpu", "igpu", or "accel"
 */
data class BackendInfo(
    val name: String,
    val description: String,
    val type: String,
)

/**
 * Model-loading progress callback. [onProgress] receives a value in 0.0–1.0
 * and is called from the loading thread. Return true to continue loading,
 * false to abort (load() will then throw).
 */
fun interface LoadProgressCallback {
    fun onProgress(progress: Float): Boolean
}

class LlamaEngine {

    private var handle: Long = 0L

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /**
     * List all available ggml backend devices (CPU, Vulkan, OpenCL, …).
     * Static — does not require a model to be loaded.
     * Parses the JSON returned by nativeListBackends().
     */
    fun availableBackends(): List<BackendInfo> {
        val raw = nativeListBackends()
        val arr = JSONArray(raw)
        return (0 until arr.length()).map { i ->
            val obj = arr.getJSONObject(i)
            BackendInfo(
                name        = obj.optString("deviceName", "Unknown"),
                description = obj.optString("backend", "unknown"),
                type        = obj.optString("type", "unknown"),
            )
        }
    }

    /**
     * Return the backend actually used for inference on the loaded model.
     * Deduced from n_gpu_layers and the first GPU device in the ggml registry.
     * Requires a model to be loaded.
     */
    fun activeBackend(): String = nativeActiveBackend(handle)

    /**
     * Load a GGUF model from [path].
     * [nGpuLayers] = 0 → CPU only; > 0 → offload layers to GPU (Vulkan on Mali).
     * [nCtx] = context window size (tokens). Defaults to 4096.
     * [nThreads] = CPU inference threads.
     *   0 (default) → pin to the number of big cores ([detectBigCoreCount]).
     *     On big.LITTLE SoCs slow cores drag the whole pool down — measured
     *     on a 2×A78+6×A55 chip, llama.cpp's own auto was up to 6× slower
     *     than big-core pinning. Falls back to llama.cpp auto if sysfs
     *     detection fails.
     *   -1 → force llama.cpp auto-detect.
     *   > 0 → explicit thread count.
     * [onProgress] — optional loading progress (0.0–1.0), called from the
     *   loading thread; return false to abort the load.
     * Throws [IllegalStateException] if the native load fails or is aborted.
     */
    fun load(
        path: String,
        nGpuLayers: Int = 0,
        nCtx: Int = 4096,
        nThreads: Int = 0,
        onProgress: LoadProgressCallback? = null,
    ) {
        val resolved = when {
            nThreads > 0 -> nThreads
            nThreads == 0 -> detectBigCoreCount()  // 0 on failure → llama auto
            else -> 0                              // -1 → llama auto
        }
        handle = nativeLoadModel(path, nGpuLayers, nCtx, resolved, onProgress)
        if (handle == 0L) {
            throw IllegalStateException("nativeLoadModel failed: $path")
        }
    }

    /**
     * Tokenize [text] and return the raw token ids.
     * Useful to measure prompt length before generating.
     */
    fun tokenize(text: String): IntArray = nativeTokenize(handle, text)

    /** Number of tokens currently occupying the KV cache (sequence 0). */
    fun kvUsed(): Int = nativeKvCacheUsedCells(handle)

    /** Interrupt an in-progress completion. Safe to call from any thread. */
    fun interrupt() = nativeInterrupt(handle)

    /**
     * Save the current KV-cache + prompt state to [path].
     * Returns the number of tokens saved, or -1 on failure.
     */
    fun saveSession(path: String): Int = nativeSaveSession(handle, path)

    /**
     * Restore a previously saved session from [path].
     * Returns the number of tokens loaded, or -1 on failure.
     */
    fun loadSession(path: String): Int = nativeLoadSession(handle, path)

    /** Release native memory. After this call the engine must not be used. */
    fun free() {
        nativeFree(handle)
        handle = 0L
    }

    /**
     * Apply the model's built-in chat template (embedded in the GGUF) to
     * [messages] and return the fully formatted prompt string.
     * Works with any model — Qwen, Llama, Mistral, etc.
     */
    fun formatChat(messages: List<ChatMessage>): String {
        val arr = JSONArray()
        for (msg in messages) {
            val obj = JSONObject()
            obj.put("role", msg.role)
            obj.put("content", msg.content)
            arr.put(obj)
        }
        return nativeFormatChat(handle, arr.toString())
    }

    /**
     * Run a completion for [prompt], streaming tokens via [callback].
     * [params] controls generation length and sampling (see [SamplingParams]).
     * This is a blocking call — run it from a background thread or
     * use the [decode] Flow extension which handles that automatically.
     */
    fun completion(
        prompt: String,
        params: SamplingParams = SamplingParams(),
        callback: TokenCallback,
    ) = nativeCompletion(
        handle, prompt,
        params.nPredict, params.temperature, params.topK, params.topP, params.minP,
        params.stopSequences.toTypedArray(),
        callback,
    )

    // ------------------------------------------------------------------
    // JNI declarations — names and types must match tensai_jni.cpp exactly
    // ------------------------------------------------------------------

    private external fun nativeListBackends(): String
    private external fun nativeActiveBackend(h: Long): String
    private external fun nativeLoadModel(
        path: String, nGpuLayers: Int, nCtx: Int, nThreads: Int,
        progressCb: LoadProgressCallback?,
    ): Long
    private external fun nativeFree(h: Long)
    private external fun nativeCompletion(
        h: Long, prompt: String,
        nPredict: Int, temperature: Float, topK: Int, topP: Float, minP: Float,
        stopSequences: Array<String>,
        cb: TokenCallback,
    )
    private external fun nativeFormatChat(h: Long, messagesJson: String): String
    private external fun nativeTokenize(h: Long, text: String): IntArray
    private external fun nativeKvCacheUsedCells(h: Long): Int
    private external fun nativeInterrupt(h: Long)
    private external fun nativeSaveSession(h: Long, path: String): Int
    private external fun nativeLoadSession(h: Long, path: String): Int

    companion object {
        init {
            System.loadLibrary("llamakt")
        }

        /**
         * Count the "big" cores of a big.LITTLE SoC: cores whose
         * cpuinfo_max_freq is at least 85% of the fastest core's.
         * The 85% cutoff separates efficiency cores (e.g. A55 at 2.0GHz
         * vs A78 at 2.4GHz = 83%) while keeping mid cores on 3-tier SoCs
         * (e.g. A78 at 2.6GHz vs X1 at 3.0GHz = 87%).
         * Returns 0 if sysfs is unreadable (caller should fall back to auto).
         */
        fun detectBigCoreCount(): Int = try {
            val freqs = java.io.File("/sys/devices/system/cpu")
                .listFiles { f -> f.name.matches(Regex("cpu\\d+")) }
                ?.mapNotNull { cpu ->
                    runCatching {
                        java.io.File(cpu, "cpufreq/cpuinfo_max_freq")
                            .readText().trim().toLong()
                    }.getOrNull()
                }
                .orEmpty()
            if (freqs.isEmpty()) 0
            else {
                val max = freqs.max()
                freqs.count { it * 100 >= max * 85 }
            }
        } catch (_: Exception) { 0 }
    }
}
