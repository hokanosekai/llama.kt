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
 * The flow runs on [Dispatchers.Default]. Cancellation automatically
 * calls [LlamaEngine.interrupt] via [awaitClose].
 */
fun LlamaEngine.decode(prompt: String): Flow<String> = callbackFlow {
    val callback = TokenCallback { token -> trySend(token) }
    completion(prompt, callback)
    close()
    awaitClose { interrupt() }
}.flowOn(Dispatchers.Default)
