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
 *
 * Defaults match llama.cpp's common_params_sampling.
 */
data class SamplingParams(
    val nPredict: Int = 512,
    val temperature: Float = 0.8f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
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
     * Throws [IllegalStateException] if the native load fails.
     */
    fun load(path: String, nGpuLayers: Int = 0, nCtx: Int = 4096) {
        handle = nativeLoadModel(path, nGpuLayers, nCtx)
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
        params.nPredict, params.temperature, params.topK, params.topP,
        callback,
    )

    // ------------------------------------------------------------------
    // JNI declarations — names and types must match tensai_jni.cpp exactly
    // ------------------------------------------------------------------

    private external fun nativeListBackends(): String
    private external fun nativeActiveBackend(h: Long): String
    private external fun nativeLoadModel(path: String, nGpuLayers: Int, nCtx: Int): Long
    private external fun nativeFree(h: Long)
    private external fun nativeCompletion(
        h: Long, prompt: String,
        nPredict: Int, temperature: Float, topK: Int, topP: Float,
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
    }
}
