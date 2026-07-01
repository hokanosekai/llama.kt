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
 * [nPredict] caps the number of generated tokens (hard stop against infinite
 * generation). Defaults to 512.
 *
 * The flow runs on [Dispatchers.Default]. Cancellation automatically
 * calls [LlamaEngine.interrupt] via [awaitClose].
 */
fun LlamaEngine.decode(prompt: String, nPredict: Int = 512): Flow<String> = callbackFlow {
    val callback = TokenCallback { token -> trySend(token) }
    completion(prompt, nPredict, callback)
    close()
    awaitClose { interrupt() }
}.flowOn(Dispatchers.Default)

/**
 * Chat-mode Flow: applies the model's built-in chat template to [messages],
 * then streams the response tokens.
 *
 * The template is read from the GGUF metadata — works with any model
 * (Qwen, Llama, Mistral, …) without hardcoding a format. EOS is emitted
 * naturally by the template, and [nPredict] acts as a hard cap.
 *
 * Usage:
 *   val messages = listOf(ChatMessage("user", "Hello, who are you?"))
 *   engine.chat(messages).collect { token -> print(token) }
 */
fun LlamaEngine.chat(messages: List<ChatMessage>, nPredict: Int = 512): Flow<String> {
    val prompt = formatChat(messages)
    return decode(prompt, nPredict)
}
