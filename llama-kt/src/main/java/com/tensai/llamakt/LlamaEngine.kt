package com.tensai.llamakt

import org.json.JSONArray
import org.json.JSONObject

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
 * [repeatPenalty]: multiplicative penalty on tokens already generated in this
 *   completion (llama.cpp `penalty_repeat`). 1.0 disables it (default, no behaviour
 *   change). Applied first in the sampler chain, before topK/topP/minP/temperature.
 *   Its window only holds tokens the sampler accepted: generated tokens, reasoning
 *   included, never the prompt.
 * [repeatLastN]: how many of the most recent generated tokens the penalty looks
 *   at (`penalty_last_n`). 0 or negative disables (this build clamps -1 to 0). Default 64.
 * [stopSequences] — strings that end generation when they appear in the
 *   output. The matched sequence is not emitted (partial matches are held
 *   back until resolved), so the stream stays clean.
 * [reasoningBudgetTokens] — cap on the tokens a thinking model may spend
 *   *inside* its reasoning block, `-1` (the default) for no cap. Reaching it
 *   does not end the generation: llama.cpp forces the model out of the block
 *   by emitting the template's own closing tag, and the model then writes its
 *   answer into what [nPredict] has left. Only applies to a [prompt] that came
 *   from [LlamaEngine.formatChat] and to a model whose chat template delimits
 *   reasoning — it is silently ignored otherwise, since there is no tag to
 *   force. Independent of [nPredict], which still caps the decode as a whole.
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
    val reasoningBudgetTokens: Int = -1,
    val repeatPenalty: Float = 1.0f,
    val repeatLastN: Int = 64,
)

/**
 * Information about a single ggml backend device.
 * [name] - ggml device id (e.g. "Vulkan0"), not a hardware name
 * [deviceDescription] - driver-reported hardware name (e.g. "Mali-G68 MC4"), empty when unknown
 * [description] — backend registry name (e.g. "Vulkan", "CPU")
 * [type] — "cpu", "gpu", "igpu", or "accel"
 */
data class BackendInfo(
    val name: String,
    val description: String,
    val type: String,
    val deviceDescription: String = "",
)

/**
 * Coarse phases of a model load, in the order [LoadProgressCallback.onStage] reports them.
 * [id] is the value the native side sends (see LOAD_STAGE_READY and common_load_stage).
 */
enum class LoadStage(val id: Int) {
    /** Reading and mapping the weights; [LoadProgressCallback.onProgress] reports 0..1 in here. */
    Weights(0),

    /** Creating the context: KV cache allocation and compute graph reservation. No progress. */
    Context(1),

    /** First decode on the freshly built context (backend warm-up). No progress. */
    Warmup(2),

    /** The model is usable. */
    Ready(3),

    ;

    companion object {
        fun fromId(id: Int): LoadStage? = entries.firstOrNull { it.id == id }
    }
}

/**
 * Model-loading progress callback. [onProgress] receives a value in 0.0-1.0
 * (weights only) and is called from the loading thread. Return true to
 * continue loading, false to abort (load() will then throw).
 *
 * [onStage] is optional on the native side: it looks the method up by name and skips stage
 * events when the implementation has none. Do not rely on a SAM lambda receiving stages:
 * depending on the Kotlin default-method mode, `onStage` can be abstract in the bytecode of
 * the lambda class or absent from it. To get stages, implement the interface in an `object`
 * and override both methods. Stage ids are [LoadStage.id]; unknown ids should be ignored.
 */
fun interface LoadProgressCallback {
    fun onProgress(progress: Float): Boolean

    fun onStage(stage: Int) {}
}

/**
 * GGUF header metadata, read without loading the weights (milliseconds even
 * for multi-GB files). See [LlamaEngine.readMetadata].
 *
 * [paramCount] is computed from the tensor infos (sum of elements), so it is
 * the real total — including e.g. per-layer embeddings on Gemma 3n.
 * [quantLabel] is a human-readable name for [fileType] ("Q4_K_M", …).
 *
 * @param headCount largest attention head count over the layers (the key can
 *   be a per-layer array). 0 when the model has no attention heads or the key
 *   is unreadable.
 * @param headCountKv largest KV head count over the layers — below [headCount]
 *   on a grouped-query model. Same 0-means-unknown rule.
 * @param kvFullElementsPerToken K+V cache elements one context token occupies
 *   across the layers whose cache is sized by `n_ctx`. **0 means the shape
 *   could not be read** — callers must fall back to their own estimate rather
 *   than treating it as "no cache".
 * @param kvSwaElementsPerToken same, for the sliding-window layers, whose
 *   cache is capped at [slidingWindow] (plus one ubatch) instead of growing
 *   with `n_ctx`. 0 when the model has none, or when which layers are windowed
 *   could not be established — in that case those layers are already counted
 *   in [kvFullElementsPerToken], which over-estimates rather than under-.
 * @param slidingWindow window size in tokens, 0 unless
 *   [kvSwaElementsPerToken] is non-zero.
 * @param kvShapeVersion which revision of the native derivation produced the
 *   three fields above — see [LlamaEngine.kvShapeVersion]. Persist it
 *   alongside them if you persist them at all; 0 only ever comes from a
 *   library too old to report one.
 */
data class GgufMetadata(
    val architecture: String,
    val name: String,
    val fileType: Int,
    val quantLabel: String,
    val contextLength: Long,
    val embeddingLength: Long,
    val blockCount: Long,
    val paramCount: Long,
    val vocabSize: Long,
    val fileSizeBytes: Long,
    /** Detected from the chat template (enable_thinking / <think> markers). */
    val supportsThinking: Boolean,
    /** Raw Jinja chat template from the header, empty when it has none. */
    val chatTemplate: String = "",
    /** The GGUF also ships a separate `tokenizer.chat_template.tool_use` template. */
    val hasToolUseTemplate: Boolean = false,
    val headCount: Long = 0,
    val headCountKv: Long = 0,
    val kvFullElementsPerToken: Long = 0,
    val kvSwaElementsPerToken: Long = 0,
    val slidingWindow: Long = 0,
    val kvShapeVersion: Long = 0,
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
                deviceDescription = obj.optString("deviceDescription", ""),
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
     * [nThreadsBatch] = CPU threads for prompt prefill (batch decode).
     *   0 (default) → with an explicit [nThreads] the same count, otherwise
     *   [ThreadPolicy]: the big cores plus half of the others, leaving one core
     *   for the UI. -1 → llama.cpp auto-detect (follows [nThreads] when that is explicit). > 0 → explicit thread count.
     * [onProgress] — optional loading progress (0.0–1.0), called from the
     *   loading thread; return false to abort the load.
     * [kvCacheType] — KV cache quantization: null → f16 (default), "q8_0" →
     *   half the cache memory (double the context in the same RAM), "q4_0" →
     *   quarter (quality risk). Applied to both K and V. Quantized V cache
     *   requires flash attention — do not combine with flashAttn = "off"
     *   (the load will fail).
     * [flashAttn] — flash attention: null → auto (llama.cpp decides), "on",
     *   "off". On GPUs without matrix cores (Mali) the generic FA path can
     *   cost instead of help — worth benchmarking per device.
     * [chatTemplate]: chat template to use instead of the GGUF's own: null or empty →
     *   the file's, "chatml" / "mistral" / "llama3" / "gemma" / "phi3" → that format, any other
     *   string → taken as a Jinja template source.
     * Throws [IllegalStateException] if the native load fails or is aborted.
     */
    fun load(
        path: String,
        nGpuLayers: Int = 0,
        nCtx: Int = 4096,
        nThreads: Int = 0,
        onProgress: LoadProgressCallback? = null,
        kvCacheType: String? = null,
        flashAttn: String? = null,
        chatTemplate: String? = null,
        nThreadsBatch: Int = 0,
    ) {
        val policy = if (nThreads == 0) {
            ThreadPolicy.resolve(ThreadPolicy.readCoreSpeeds())
        } else {
            ThreadCounts(0, 0)
        }
        val resolved = when {
            nThreads > 0 -> nThreads
            nThreads == 0 -> policy.decode  // 0 on failure → llama auto
            else -> 0                       // -1 → llama auto
        }
        val resolvedBatch = when {
            nThreadsBatch > 0 -> nThreadsBatch
            nThreadsBatch < 0 -> 0
            nThreads > 0 -> nThreads
            nThreads < 0 -> 0
            else -> policy.batch
        }
        handle = nativeLoadModel(
            path, nGpuLayers, nCtx, resolved, resolvedBatch, onProgress, kvCacheType, flashAttn, chatTemplate,
        )
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
     *
     * [enableThinking] — thinking-capable models (Qwen3, …) reason inside
     *   <think>…</think> before answering. true (default) keeps that;
     *   false renders the template with thinking disabled: faster first
     *   visible token, at some quality cost on hard prompts. No effect on
     *   models without a thinking template.
     *
     * Also records, for this engine, the chat format and parser llama.cpp
     * derived from the template — that is what lets [completion] fill a
     * [ChatParseCallback] when handed back this exact prompt.
     */
    fun formatChat(messages: List<ChatMessage>, enableThinking: Boolean = true): String =
        nativeFormatChat(handle, ToolJson.messages(messages).toString(), enableThinking, "", "", false)

    /**
     * [formatChat] with [tools] rendered into the prompt through the model's chat template, which
     * is what makes the model able to call them. [messages] may carry assistant [ChatMessage.toolCalls]
     * and role "tool" messages, to replay earlier calls with their results.
     *
     * A template that cannot render tools does not fail: the prompt is built without them and
     * [FormattedChat.toolsIgnored] says so, since the model then never hears of the tools.
     * Empty [tools] is plain [formatChat].
     *
     * [toolChoice]: see [ToolChoice]. [parallelToolCalls]: whether one reply may hold several
     * calls. It changes the parser llama.cpp derives (how many calls it accepts), so it must match
     * what the caller is ready to execute.
     *
     * @throws IllegalArgumentException when a replayed [ToolCall.argumentsJson] is not a JSON object.
     *
     * After [completion] on this prompt, [lastChatResult] holds the parsed tool calls.
     */
    fun formatChatWithTools(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        toolChoice: ToolChoice = ToolChoice.Auto,
        enableThinking: Boolean = true,
        parallelToolCalls: Boolean = true,
    ): FormattedChat {
        val toolsJson = if (tools.isEmpty()) "" else ToolJson.tools(tools).toString()
        val prompt = nativeFormatChat(
            handle, ToolJson.messages(messages).toString(), enableThinking, toolsJson,
            if (tools.isEmpty()) "" else toolChoice.wire, parallelToolCalls,
        )
        val ignored = ToolJson.toolsIgnored(nativeChatResult(handle))
        return FormattedChat(prompt, ignored)
    }

    /**
     * The reply of the last [completion], as llama.cpp's chat parser (the one derived from the
     * template, same as [ChatParseCallback]) split it into answer, reasoning and tool calls.
     * Final, not cumulative. Read it right after [completion] returns, before any [formatChat],
     * [formatChatWithTools] or warm-up: there is one slot per engine and those overwrite it.
     * Ignore it when [completion] returned 0 or less (nothing was generated, or the native side
     * failed): the result is then cleared or partial.
     *
     * Null when there is nothing to report: no completion since the last [formatChat] /
     * [formatChatWithTools], a prompt that did not come from them, or a template that could not be
     * rendered with jinja. A reply the parser rejects is not null: see [ChatResult.toolParseFailed].
     *
     * Streaming: [TokenCallback] always gets the raw text, a tool call included. For a stream that
     * never shows call text, display the [ChatParseCallback] content instead: the parser holds a
     * call back as it is written and keeps it out of the content. Calls themselves only exist here,
     * in the final result.
     */
    fun lastChatResult(): ChatResult? = ToolJson.result(nativeChatResult(handle))

    /**
     * Run a completion for [prompt], streaming tokens via [callback].
     * [params] controls generation length and sampling (see [SamplingParams]).
     * This is a blocking call — run it from a background thread or
     * use the [decode] Flow extension which handles that automatically.
     *
     * [chatParseCallback] — optional, and only useful when [prompt] came from
     *   [formatChat]: streams the same reply already split into answer and
     *   reasoning by llama.cpp's chat parser. See [ChatParseCallback] for when
     *   it does *not* fire (it is best-effort, and callers need a fallback).
     *
     * @return how many tokens the decode sampled. Compare it against
     *   [SamplingParams.nPredict] to tell a reply the model ended itself from
     *   one that ran out of budget — this is the count llama.cpp itself kept,
     *   and **not** the number of [TokenCallback] invocations, which is a
     *   different (smaller, and unrelated) number: tokens are held back while a
     *   UTF-8 sequence is incomplete or a stop sequence is half-matched, so one
     *   callback can carry several tokens. `0` when the completion could not
     *   start at all.
     *
     *   **Negative means the native side failed**, and is not a count: a C++
     *   exception escaped the decode and `tensai_jni.cpp`'s JNI guard turned it
     *   into a value rather than letting it kill the process. Callers must test
     *   for that before comparing against [SamplingParams.nPredict] — `-1` is
     *   below every cap, so a comparison would read a crashed decode as a reply
     *   the model chose to end. Any tokens already streamed to [callback] stay
     *   streamed; what is lost is the ending, not the text.
     */
    fun completion(
        prompt: String,
        params: SamplingParams = SamplingParams(),
        callback: TokenCallback,
        chatParseCallback: ChatParseCallback? = null,
    ) = nativeCompletion(
        handle, prompt,
        params.nPredict, params.reasoningBudgetTokens,
        params.temperature, params.topK, params.topP, params.minP,
        params.repeatPenalty, params.repeatLastN,
        params.stopSequences.toTypedArray(),
        callback,
        chatParseCallback,
    )

    /**
     * Decode [prompt] into the KV cache without generating anything, [completion] up to
     * and including its prefill, and not a token further. Blocking, like [completion]: it
     * holds the calling thread for as long as the prefill takes (minutes, for a long
     * prompt on a phone), and [interrupt] ends it between chunks.
     *
     * Goes through the same prefix reuse as [completion]: whatever the cache already holds
     * of [prompt]'s leading tokens is kept, the rest is dropped and decoded. The cache is
     * then bookkept as a completion would have left it, so a [completion] that follows with
     * a prompt starting with [prompt] reuses all of it, which is the point, see
     * `nativePrefill` in `tensai_jni.cpp` for why a caller wants that.
     *
     * An interrupted prefill keeps the chunks it finished: the cache holds a valid prefix of
     * [prompt] and [kvUsed] says how long.
     *
     * @return how many tokens the cache holds afterwards ([kvUsed]), also when interrupted.
     *   `0` when the prompt was declined without touching the cache (it does not fit the
     *   context window, or the model is encoder-decoder). **Negative means the native side
     *   failed**; the cache is still consistent, with whatever was decoded before the failure.
     */
    fun prefill(prompt: String): Int = nativePrefill(handle, prompt)

    // ------------------------------------------------------------------
    // JNI declarations — names and types must match tensai_jni.cpp exactly
    // ------------------------------------------------------------------

    private external fun nativeListBackends(): String
    private external fun nativeActiveBackend(h: Long): String
    private external fun nativeLoadModel(
        path: String, nGpuLayers: Int, nCtx: Int, nThreads: Int, nThreadsBatch: Int,
        progressCb: LoadProgressCallback?, kvCacheType: String?, flashAttn: String?, chatTemplate: String?,
    ): Long
    private external fun nativeFree(h: Long)
    private external fun nativeCompletion(
        h: Long, prompt: String,
        nPredict: Int, reasoningBudgetTokens: Int,
        temperature: Float, topK: Int, topP: Float, minP: Float,
        repeatPenalty: Float, repeatLastN: Int,
        stopSequences: Array<String>,
        cb: TokenCallback,
        cbChat: ChatParseCallback?,
    ): Int
    private external fun nativePrefill(h: Long, prompt: String): Int
    private external fun nativeFormatChat(
        h: Long, messagesJson: String, enableThinking: Boolean, toolsJson: String, toolChoice: String,
        parallelToolCalls: Boolean,
    ): String
    private external fun nativeChatResult(h: Long): String
    private external fun nativeTokenize(h: Long, text: String): IntArray
    private external fun nativeKvCacheUsedCells(h: Long): Int
    private external fun nativeInterrupt(h: Long)
    private external fun nativeSaveSession(h: Long, path: String): Int
    private external fun nativeLoadSession(h: Long, path: String): Int

    companion object {
        init {
            System.loadLibrary("llamakt")
        }

        // llama_ftype enum values → human labels (common subset)
        private val FTYPE_LABELS = mapOf(
            0 to "F32", 1 to "F16", 2 to "Q4_0", 3 to "Q4_1", 7 to "Q8_0",
            8 to "Q5_0", 9 to "Q5_1", 10 to "Q2_K", 11 to "Q3_K_S",
            12 to "Q3_K_M", 13 to "Q3_K_L", 14 to "Q4_K_S", 15 to "Q4_K_M",
            16 to "Q5_K_S", 17 to "Q5_K_M", 18 to "Q6_K", 24 to "IQ4_NL",
            25 to "IQ3_S", 26 to "IQ3_M", 30 to "BF16",
        )

        /**
         * Read the GGUF header of [path] without loading the model.
         * Fast (header only) — safe to call on the UI flow for an import
         * screen. Returns null if the file is not a readable GGUF.
         */
        fun readMetadata(path: String): GgufMetadata? {
            val raw = nativeReadGgufMetadata(path) ?: return null
            return try {
                val obj = JSONObject(raw)
                val ftype = obj.optInt("file_type", -1)
                GgufMetadata(
                    architecture    = obj.optString("architecture", "unknown"),
                    name            = obj.optString("name", ""),
                    fileType        = ftype,
                    quantLabel      = FTYPE_LABELS[ftype] ?: "ftype_$ftype",
                    contextLength   = obj.optLong("context_length", 0),
                    embeddingLength = obj.optLong("embedding_length", 0),
                    blockCount      = obj.optLong("block_count", 0),
                    paramCount      = obj.optLong("param_count", 0),
                    vocabSize       = obj.optLong("vocab_size", 0),
                    fileSizeBytes   = obj.optLong("file_size_bytes", 0),
                    supportsThinking = obj.optBoolean("supports_thinking", false),
                    chatTemplate    = obj.optString("chat_template", ""),
                    hasToolUseTemplate = obj.optBoolean("tool_use_template", false),
                    headCount       = obj.optLong("head_count", 0),
                    headCountKv     = obj.optLong("head_count_kv", 0),
                    kvFullElementsPerToken = obj.optLong("kv_full_elements_per_token", 0),
                    kvSwaElementsPerToken  = obj.optLong("kv_swa_elements_per_token", 0),
                    slidingWindow          = obj.optLong("sliding_window", 0),
                    kvShapeVersion         = obj.optLong("kv_shape_version", 0),
                )
            } catch (_: Exception) { null }
        }

        @JvmStatic
        private external fun nativeReadGgufMetadata(path: String): String?

        /**
         * Which revision of the native KV-shape derivation this build carries
         * — the same number [readMetadata] stamps into
         * [GgufMetadata.kvShapeVersion], but answerable without a file.
         *
         * For callers that persist a model's KV shape: store this alongside
         * it, and re-read whatever was stored under a *different* number. The
         * comparison is equality, not ordering — the derivation is versioned
         * so a change can be detected, and a change that lowers a number
         * matters as much as one that raises it. Treat a version you don't
         * recognise as a difference, not as an error.
         *
         * Defined in `tensai_jni.cpp` next to the derivation itself, so it
         * cannot drift from the formula it describes.
         */
        fun kvShapeVersion(): Long = nativeKvShapeVersion()

        /**
         * llama.cpp's own system info string, e.g.
         * `CPU : NEON = 1 | ARM_FMA = 1 | DOTPROD = 1 | ...`: the CPU features ggml-cpu was
         * built with and detected at runtime. Static, no model needed. Empty when the native
         * side failed.
         */
        fun systemInfo(): String = nativeSystemInfo()

        /**
         * The `general.architecture` values the embedded llama.cpp can load, read from its own
         * architecture table. Static, no model needed. Empty only when the native side failed:
         * callers must treat an empty set as "unknown", not as "nothing is supported".
         */
        fun supportedArchitectures(): Set<String> =
            nativeSupportedArchitectures().lineSequence().filter { it.isNotBlank() }.toSet()

        /**
         * The first error line llama.cpp logged during the most recent [load] (e.g.
         * `llama_model_load: error loading model: ... unknown model architecture: 'x'`), empty
         * when it logged none. Meant to be read right after [load] threw.
         */
        fun lastLoadError(): String = nativeLastLoadError()

        @JvmStatic
        private external fun nativeSupportedArchitectures(): String

        @JvmStatic
        private external fun nativeLastLoadError(): String

        @JvmStatic
        private external fun nativeSystemInfo(): String

        @JvmStatic
        private external fun nativeKvShapeVersion(): Long

        /**
         * Count the "big" cores of a big.LITTLE SoC, see [ThreadPolicy.bigCoreCount].
         * Returns 0 if sysfs is unreadable (caller should fall back to auto).
         */
        fun detectBigCoreCount(): Int = ThreadPolicy.bigCoreCount(ThreadPolicy.readCoreSpeeds())
    }
}
