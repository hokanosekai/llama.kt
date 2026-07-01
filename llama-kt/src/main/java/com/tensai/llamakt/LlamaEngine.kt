package com.tensai.llamakt

class LlamaEngine {

    private var handle: Long = 0L

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /**
     * Load a GGUF model from [path].
     * [nGpuLayers] = 0 → CPU only; > 0 → offload to GPU via OpenCL.
     * Throws [IllegalStateException] if the native load fails.
     */
    fun load(path: String, nGpuLayers: Int = 0) {
        handle = nativeLoadModel(path, nGpuLayers)
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
     * Run a completion for [prompt], streaming tokens via [callback].
     * This is a blocking call — run it from a background thread or
     * use the [decode] Flow extension which handles that automatically.
     */
    fun completion(prompt: String, callback: TokenCallback) =
        nativeCompletion(handle, prompt, callback)

    // ------------------------------------------------------------------
    // JNI declarations — names and types must match tensai_jni.cpp exactly
    // ------------------------------------------------------------------

    private external fun nativeLoadModel(path: String, nGpuLayers: Int): Long
    private external fun nativeFree(h: Long)
    private external fun nativeCompletion(h: Long, prompt: String, cb: TokenCallback)
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
