package com.tensai.llamakt

import org.json.JSONArray
import org.json.JSONObject

/**
 * A tool the model may call. [parametersJson] is the JSON Schema of the arguments, as a string.
 */
data class ToolDefinition(
    val name: String,
    val description: String,
    val parametersJson: String = """{"type":"object","properties":{}}""",
)

/**
 * A tool call: parsed from a reply, or carried by an assistant [ChatMessage] to replay it.
 * [argumentsJson] is a JSON object serialized as a string, as the model wrote it: it may not
 * validate against the tool's schema, checking it is the caller's job.
 */
data class ToolCall(val id: String, val name: String, val argumentsJson: String)

/** Whether and how the model may call tools. [wire] is the value llama.cpp's template layer reads. */
enum class ToolChoice(val wire: String) {
    /** The model decides. */
    Auto("auto"),

    /** Tools are described in the prompt but the model must not call them. */
    None("none"),

    /** The model must call a tool. */
    Required("required"),
}

/**
 * A single message in a chat conversation.
 *
 * [toolCalls]: calls an assistant message made, so a later turn can replay them next to their results.
 * [toolCallId]: for role "tool", the [ToolCall.id] this message answers.
 * [reasoningContent]: the reasoning an assistant message came with, sent as `reasoning_content`. Whether it is
 *   rendered is the template's call (Qwen3 keeps it after the last user message only). Empty sends nothing.
 */
data class ChatMessage(
    val role: String,
    val content: String,
    val toolCalls: List<ToolCall> = emptyList(),
    val toolCallId: String? = null,
    val reasoningContent: String = "",
)

/**
 * [LlamaEngine.formatChatWithTools] output: the prompt, and whether the tools made it into it.
 * [toolsIgnored]: the template cannot render tools, so [prompt] was built without them.
 */
data class FormattedChat(val prompt: String, val toolsIgnored: Boolean)

/**
 * What llama.cpp's chat parser made of the last completion, see [LlamaEngine.lastChatResult].
 *
 * [content]: the answer, markers and tool-call text stripped. The raw reply when [toolParseFailed].
 * [toolCalls]: calls found in the reply. Empty when there are none, when the reply was
 *   interrupted or [truncated] (its last call would be half-written) or when the parse failed.
 * [toolsIgnored]: the prompt was rendered without the tools, see [FormattedChat.toolsIgnored].
 * [toolParseFailed]: the parser rejected the reply (a malformed call, typically). The reply is in
 *   [content] as written and no call is reported. Nothing is thrown for it.
 * [truncated]: the reply stopped on the token cap or a full context window, not on its own end.
 *   It was parsed leniently, so reasoning is still split from the answer.
 */
data class ChatResult(
    val content: String,
    val reasoningContent: String,
    val toolCalls: List<ToolCall>,
    val toolsIgnored: Boolean,
    val toolParseFailed: Boolean,
    val truncated: Boolean = false,
)

/**
 * The JSON shapes exchanged with the native side (OpenAI-style, what llama.cpp's chat layer
 * reads and writes). Kept apart from [LlamaEngine] so it runs on the JVM, without the library.
 */
internal object ToolJson {

    /**
     * @throws IllegalArgumentException when a replayed [ToolCall.argumentsJson] is not a JSON
     *   object: native would otherwise fail inside the template and silently fall back to a
     *   renderer that has no tools.
     */
    fun messages(messages: List<ChatMessage>): JSONArray {
        val arr = JSONArray()
        for (msg in messages) {
            val obj = JSONObject()
            obj.put("role", msg.role)
            obj.put("content", msg.content)
            if (msg.toolCalls.isNotEmpty()) {
                val calls = JSONArray()
                for (c in msg.toolCalls) {
                    requireJsonObject(c)
                    calls.put(
                        JSONObject()
                            .put("id", c.id)
                            .put("type", "function")
                            .put("function", JSONObject().put("name", c.name).put("arguments", c.argumentsJson)),
                    )
                }
                obj.put("tool_calls", calls)
            }
            if (msg.toolCallId != null) obj.put("tool_call_id", msg.toolCallId)
            if (msg.reasoningContent.isNotEmpty()) obj.put("reasoning_content", msg.reasoningContent)
            arr.put(obj)
        }
        return arr
    }

    private fun requireJsonObject(c: ToolCall) {
        try {
            JSONObject(c.argumentsJson)
        } catch (e: Exception) {
            throw IllegalArgumentException(
                "ToolCall '${c.name}' (id=${c.id}): argumentsJson is not a JSON object: ${e.message}",
            )
        }
    }

    /** `[{"type":"function","function":{name, description, parameters}}]`. */
    fun tools(tools: List<ToolDefinition>): JSONArray {
        val arr = JSONArray()
        for (t in tools) {
            // Unparseable schema: an empty one, so a bad tool degrades instead of failing the whole prompt.
            val params = try { JSONObject(t.parametersJson) } catch (_: Exception) { JSONObject() }
            arr.put(
                JSONObject()
                    .put("type", "function")
                    .put(
                        "function",
                        JSONObject().put("name", t.name).put("description", t.description).put("parameters", params),
                    ),
            )
        }
        return arr
    }

    /**
     * Reads the `nativeChatResult` JSON. Null when it is not JSON, or when no completion has
     * produced a result (`has_result` false).
     */
    fun result(raw: String): ChatResult? = try {
        val obj = JSONObject(raw)
        if (!obj.optBoolean("has_result", false)) {
            null
        } else {
            val arr = obj.optJSONArray("tool_calls") ?: JSONArray()
            ChatResult(
                content = obj.optString("content", ""),
                reasoningContent = obj.optString("reasoning_content", ""),
                toolCalls = (0 until arr.length()).map { i ->
                    val c = arr.getJSONObject(i)
                    ToolCall(c.optString("id", ""), c.optString("name", ""), c.optString("arguments", "{}"))
                },
                toolsIgnored = obj.optBoolean("tools_ignored", false),
                toolParseFailed = obj.optBoolean("tool_parse_failed", false),
                truncated = obj.optBoolean("truncated", false),
            )
        }
    } catch (_: Exception) { null }

    /** `tools_ignored` of the `nativeChatResult` JSON, readable before any completion. */
    fun toolsIgnored(raw: String): Boolean =
        try { JSONObject(raw).optBoolean("tools_ignored", false) } catch (_: Exception) { false }
}
