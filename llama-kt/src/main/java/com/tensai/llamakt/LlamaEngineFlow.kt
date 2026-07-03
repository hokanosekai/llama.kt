package com.tensai.llamakt

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn

/**
 * Stream completion tokens as a [Flow].
 *
 * Usage:
 *   engine.load("/sdcard/model.gguf")
 *   engine.decode("Hello").collect { token -> print(token) }
 *
 * [params] controls generation length and sampling — see [SamplingParams].
 * The nPredict cap is a hard stop against infinite generation.
 *
 * The flow runs on [Dispatchers.Default]. Cancellation automatically
 * calls [LlamaEngine.interrupt] via [awaitClose].
 */
fun LlamaEngine.decode(
    prompt: String,
    params: SamplingParams = SamplingParams(),
): Flow<String> = callbackFlow {
    val callback = TokenCallback { token -> trySend(token) }
    completion(prompt, params, callback)
    close()
    awaitClose { interrupt() }
}.flowOn(Dispatchers.Default)

/**
 * Chat-mode Flow: applies the model's built-in chat template to [messages],
 * then streams the response tokens.
 *
 * The template is read from the GGUF metadata — works with any model
 * (Qwen, Llama, Mistral, …) without hardcoding a format. EOS is emitted
 * naturally by the template, and [SamplingParams.nPredict] acts as a hard cap.
 *
 * Usage:
 *   val messages = listOf(ChatMessage("user", "Hello, who are you?"))
 *   engine.chat(messages).collect { token -> print(token) }
 */
fun LlamaEngine.chat(
    messages: List<ChatMessage>,
    params: SamplingParams = SamplingParams(),
    enableThinking: Boolean = true,
): Flow<String> {
    val prompt = formatChat(messages, enableThinking)
    return decode(prompt, params)
}
