package com.tensai.llamakt

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ToolJsonTest {

    @Test
    fun plainMessageKeepsTheOldShape() {
        val arr = ToolJson.messages(listOf(ChatMessage("user", "salut")))
        val obj = arr.getJSONObject(0)
        assertEquals(setOf("role", "content"), obj.keySet())
        assertEquals("salut", obj.getString("content"))
    }

    @Test
    fun assistantToolCallsAreOpenAiShaped() {
        val msg = ChatMessage("assistant", "", toolCalls = listOf(ToolCall("call_1", "datetime", """{"tz":"UTC"}""")))
        val call = ToolJson.messages(listOf(msg)).getJSONObject(0).getJSONArray("tool_calls").getJSONObject(0)
        assertEquals("call_1", call.getString("id"))
        assertEquals("function", call.getString("type"))
        assertEquals("datetime", call.getJSONObject("function").getString("name"))
        // arguments stay a string, llama.cpp's parser wants it that way
        assertEquals("""{"tz":"UTC"}""", call.getJSONObject("function").getString("arguments"))
    }

    @Test
    fun toolMessageCarriesItsCallId() {
        val obj = ToolJson.messages(listOf(ChatMessage("tool", "12:00", toolCallId = "call_1"))).getJSONObject(0)
        assertEquals("tool", obj.getString("role"))
        assertEquals("call_1", obj.getString("tool_call_id"))
        assertFalse(obj.has("tool_calls"))
    }

    @Test
    fun reasoningIsSentOnlyWhenThereIsSome() {
        val arr = ToolJson.messages(
            listOf(
                ChatMessage("assistant", "", reasoningContent = "let me think"),
                ChatMessage("assistant", "hi"),
            ),
        )
        assertEquals("let me think", arr.getJSONObject(0).getString("reasoning_content"))
        assertFalse(arr.getJSONObject(1).has("reasoning_content"))
    }

    @Test
    fun reasoningSurvivesAJsonRoundTrip() {
        val text = "say \"hi\"\nthen \\ done, \u00e9\u00e8 \u65e5\u672c"
        val arr = ToolJson.messages(listOf(ChatMessage("assistant", "", reasoningContent = text)))
        assertEquals(text, org.json.JSONArray(arr.toString()).getJSONObject(0).getString("reasoning_content"))
    }

    @Test
    fun toolsBecomeFunctionDefinitions() {
        val tools = ToolJson.tools(
            listOf(ToolDefinition("calc", "does maths", """{"type":"object","properties":{"expr":{"type":"string"}}}""")),
        )
        val fn = tools.getJSONObject(0).getJSONObject("function")
        assertEquals("function", tools.getJSONObject(0).getString("type"))
        assertEquals("calc", fn.getString("name"))
        assertEquals("string", fn.getJSONObject("parameters").getJSONObject("properties").getJSONObject("expr").getString("type"))
    }

    @Test
    fun badSchemaDegradesToEmptyObject() {
        val fn = ToolJson.tools(listOf(ToolDefinition("t", "d", "not json"))).getJSONObject(0).getJSONObject("function")
        assertEquals(0, fn.getJSONObject("parameters").length())
    }

    @Test
    fun resultWithCalls() {
        val raw = JSONObject(
            """{"tools_ignored":false,"has_result":true,"tool_parse_failed":false,"content":"ok",
               "reasoning_content":"hmm","tool_calls":[{"id":"call_ab","name":"calc","arguments":"{\"expr\":\"1+1\"}"}]}""",
        ).toString()
        val r = ToolJson.result(raw)!!
        assertEquals("ok", r.content)
        assertEquals("hmm", r.reasoningContent)
        assertEquals(listOf(ToolCall("call_ab", "calc", """{"expr":"1+1"}""")), r.toolCalls)
        assertFalse(r.toolParseFailed)
    }

    @Test
    fun resultFlagsFailureAndIgnoredTools() {
        val r = ToolJson.result("""{"has_result":true,"tool_parse_failed":true,"tools_ignored":true,"content":"<tool_call>{oops"}""")!!
        assertTrue(r.toolParseFailed)
        assertTrue(r.toolsIgnored)
        assertEquals("<tool_call>{oops", r.content)
        assertTrue(r.toolCalls.isEmpty())
    }

    @Test
    fun noResultYet() {
        assertNull(ToolJson.result("""{"has_result":false}"""))
        assertNull(ToolJson.result("{}"))
        assertNull(ToolJson.result("garbage"))
        assertTrue(ToolJson.toolsIgnored("""{"tools_ignored":true}"""))
        assertFalse(ToolJson.toolsIgnored("garbage"))
    }

    @Test
    fun truncatedFlagIsRead() {
        val r = ToolJson.result("""{"has_result":true,"truncated":true,"content":"x"}""")!!
        assertTrue(r.truncated)
        assertFalse(ToolJson.result("""{"has_result":true}""")!!.truncated)
    }

    @Test
    fun toolChoiceWireValues() {
        assertEquals(listOf("auto", "none", "required"), ToolChoice.entries.map { it.wire })
    }

    @Test
    fun invalidArgumentsAreRejected() {
        for (bad in listOf("", "not json", "{\"a\":", "[1,2")) {
            val msg = ChatMessage("assistant", "", toolCalls = listOf(ToolCall("c1", "calc", bad)))
            try {
                ToolJson.messages(listOf(msg))
                org.junit.Assert.fail("accepted: $bad")
            } catch (e: IllegalArgumentException) {
                assertTrue(e.message!!.contains("calc"))
            }
        }
    }

    @Test
    fun awkwardStringsRoundTrip() {
        val nasty = "q\"uote \\ back\nnl\ttab \u2028 \uD83D\uDE80 \u0001\u001f end"
        val args = JSONObject().put("expr", nasty).toString()
        val msgs = listOf(
            ChatMessage("assistant", nasty, toolCalls = listOf(ToolCall("c1", "calc", args))),
            ChatMessage("tool", nasty, toolCallId = "c1"),
        )
        // through the serialized form, as native receives it
        val back = org.json.JSONArray(ToolJson.messages(msgs).toString())
        val first = back.getJSONObject(0)
        assertEquals(nasty, first.getString("content"))
        val argStr = first.getJSONArray("tool_calls").getJSONObject(0).getJSONObject("function").get("arguments")
        assertTrue(argStr is String)
        assertEquals(args, argStr)
        assertEquals(nasty, JSONObject(argStr as String).getString("expr"))
        assertEquals(nasty, back.getJSONObject(1).getString("content"))

        // and the result direction
        val raw = JSONObject()
            .put("has_result", true).put("content", nasty)
            .put("tool_calls", org.json.JSONArray().put(JSONObject().put("id", "c1").put("name", "calc").put("arguments", args)))
            .toString()
        val r = ToolJson.result(raw)!!
        assertEquals(nasty, r.content)
        assertEquals(args, r.toolCalls[0].argumentsJson)
    }
}
