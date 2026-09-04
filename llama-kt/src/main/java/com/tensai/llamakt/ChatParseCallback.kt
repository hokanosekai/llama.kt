package com.tensai.llamakt

/**
 * Receives the model's reply as llama.cpp's own chat parser splits it, while it streams.
 *
 * [TokenCallback] gives you the raw text the model produced, markers and all. This gives you the
 * same text already split into the answer and the reasoning that led to it — using the parser
 * llama.cpp generated from the model's own chat template, so it covers every convention that
 * template can express (`<think>…</think>`, gpt-oss/harmony `<|channel|>…`, Gemma's, …) instead of
 * hardcoding one.
 *
 * Pass it to [LlamaEngine.completion] alongside the [TokenCallback]. Both fire: the raw stream is
 * untouched, this is strictly extra.
 *
 * Contract:
 * - Only fires when the prompt came from [LlamaEngine.formatChat] on the same engine — a raw
 *   completion has no chat template behind it, so there is nothing to parse it with. It also never
 *   fires when the template could not be rendered with jinja (llama.kt falls back to a legacy path
 *   that reports no format), or when the model's output does not match the format its template
 *   advertised. **Callers must have a fallback for "this never fired".**
 * - Both arguments are *cumulative*, not deltas: each call carries the whole reply resolved so far.
 *   The parser is not incremental and a partial parse can retract text it had already attributed,
 *   so a delta stream would not be sound.
 * - Fires at most once per sampled token, and is skipped entirely when a token changes neither
 *   value (which happens while the parser holds text back on an ambiguous marker).
 * - The last call for a completion carries the fully resolved reply.
 */
fun interface ChatParseCallback {
    /**
     * @param content the answer, markers stripped.
     * @param reasoningContent the reasoning that preceded it, markers stripped. Empty when the
     *   model produced none (or when the template has no notion of reasoning at all).
     */
    fun onChatParse(content: String, reasoningContent: String)
}
