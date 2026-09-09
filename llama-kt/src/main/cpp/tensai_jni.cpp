/**
 * tensai_jni.cpp — JNI bridge for com.tensai.llamakt.LlamaEngine
 *
 * Entry points:
 *   nativeLoadModel, nativeFree, nativeCompletion,
 *   nativeFormatChat, nativeTokenize, nativeKvCacheUsedCells,
 *   nativeInterrupt, nativeSaveSession, nativeLoadSession,
 *   nativeListBackends, nativeActiveBackend,
 *   nativeReadGgufMetadata, nativeKvShapeVersion
 *
 * KV used-cells strategy: this version of llama.cpp (b9769) exposes no
 * public llama_kv_self_used_cells() or llama_kv_self_n_tokens(). The
 * closest public measure is llama_memory_seq_pos_max(mem, 0) + 1, which
 * equals n_past for sequence 0. We prefer completion->n_past directly
 * when available (always alive after loadModel succeeds).
 */

#include <jni.h>
#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <android/log.h>

#include "rn-llama.h"
#include "rn-completion.h"
#include "rn-common.hpp"
#include "common/common.h"
#include "common/chat.h"
#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <sys/stat.h>

#define LOG_TAG "TensaiJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ---------------------------------------------------------------------------
// KV-cache shape derivation version
// ---------------------------------------------------------------------------
// Stamped onto every kv_full_elements_per_token / kv_swa_elements_per_token /
// sliding_window triple this file derives (see the "KV-cache shape" block in
// nativeReadGgufMetadata), and readable on its own through
// nativeKvShapeVersion() without a file to read.
//
// It exists so a caller that *persists* those numbers can tell which
// derivation produced the ones it already has. The formula lives here, so this
// file is the only thing that knows when it changed — a version constant on
// the Kotlin or app side would be a number somebody has to remember to bump
// while editing C++ in a submodule, and forgetting once means the stale
// numbers are never noticed.
//
// **Bump this whenever the derivation below changes in a way that can move a
// number**, including a fix that only affects some architectures. Callers are
// expected to compare, not to interpret: any difference means "re-read", so an
// unrecognised (e.g. newer) version must stay a normal case for them.
//
//   1 — first derivation (per-layer n_head_kv * head dim, both halves).
//   2 — the V half charged at n_embd_v_gqa_max() for every layer, on the
//       belief that llama.cpp transposes the V cache whenever flash attention
//       is off. It does, but the cache is built before the AUTO setting is
//       resolved, so the load path this file serves never lands there — see
//       version 3.
//   3 — the V half charged back at the layer's own n_embd_v_gqa(il), and
//       gemma3n's twenty cache-owning layers honoured the way gemma4's
//       already were. Version 2 over-counted the windowed half of every
//       model with a smaller SWA head dimension than its full-attention one
//       (Gemma 4: 9216 elements/token instead of 6144), and versions 1 and 2
//       both charged gemma3n for all of its blocks instead of the first 20.
//
// Versions 1 and 2 both shipped before this constant existed, so neither was
// ever reported; a caller holding numbers with no version recorded cannot
// tell which of the two it has and must re-read.
#define TENSAI_KV_SHAPE_VERSION 3

// ---------------------------------------------------------------------------
// Log redirection — llama.cpp/ggml log to stderr by default, which is lost on
// Android. Forward everything (including GGML_ABORT/assert messages) to
// logcat so native crashes are diagnosable from `adb logcat`.
// ---------------------------------------------------------------------------

static void tensai_log_callback(lm_ggml_log_level level, const char* text, void* /* user_data */) {
    int prio;
    switch (level) {
        case LM_GGML_LOG_LEVEL_ERROR: prio = ANDROID_LOG_ERROR; break;
        case LM_GGML_LOG_LEVEL_WARN:  prio = ANDROID_LOG_WARN;  break;
        case LM_GGML_LOG_LEVEL_DEBUG: prio = ANDROID_LOG_DEBUG; break;
        default:                      prio = ANDROID_LOG_INFO;  break;
    }
    __android_log_print(prio, "llama.cpp", "%s", text);
}

// GGML_ABORT/GGML_ASSERT bypass the log callback and print to stderr before
// calling abort() — hook the abort callback so the assert message (file:line)
// reaches logcat before the process dies.
static void tensai_abort_callback(const char* message) {
    __android_log_print(ANDROID_LOG_FATAL, "llama.cpp", "GGML ABORT: %s", message);
}

extern "C" JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM* /* vm */, void* /* reserved */) {
    llama_log_set(tensai_log_callback, nullptr);
    lm_ggml_log_set(tensai_log_callback, nullptr);
    lm_ggml_set_abort_callback(tensai_abort_callback);
    return JNI_VERSION_1_6;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// C++ exception barrier for the JNI boundary (TEN-91).
//
// A C++ exception is not a Java exception. Unwinding out of a native frame
// into the JVM's has no handler to land in, so it runs off the top and
// std::terminate() kills the process — no stack trace, no logcat line of our
// own, nothing a Kotlin `try` or `runCatching` can see, because there is no
// Throwable to catch. The app is simply gone.
//
// Nearly everything below can throw. nlohmann::json's default error handler
// raises type_error.316 on any string that isn't valid UTF-8, and the GGUF
// reader does not validate the strings it hands out (gguf.cpp reads a length
// and that many bytes, no more), so a corrupted `general.name` in a header was
// enough to take the process down from nativeReadGgufMetadata — including on
// the start-up path that re-reads every stored model, which turned one damaged
// file into a crash on every launch. llama.cpp's loaders throw
// std::runtime_error, and every std::string/std::vector here can throw
// std::bad_alloc.
//
// So each entry point that can allocate or call into llama.cpp is a thin
// wrapper: the work lives in a `*_impl` and the exported symbol runs it inside
// jni_guard(), which logs the escape and answers with the failure value the
// Kotlin side already handles — nullptr, 0, -1, "". Deliberately no new Java
// exception class: the callers that already treat "the native side said no" as
// recoverable now cover "the native side threw" for free, with no new catch to
// add and no contract to change.
//
// The three that are not wrapped read a field and return it —
// nativeKvShapeVersion (a compile-time constant), nativeKvCacheUsedCells and
// nativeInterrupt. Nothing in them allocates, so there is nothing to catch;
// anything added to one of them wants a wrapper.
//
// `on_throw` is a callable, not a value, so a fallback may itself be a JNI call
// (NewStringUTF, NewIntArray) rather than only a scalar. It must return the
// same type as `body`.
template <typename F, typename G>
static auto jni_guard(const char* fn, F&& body, G&& on_throw) -> decltype(body()) {
    try {
        return body();
    } catch (const std::exception& e) {
        LOGE("%s: C++ exception at the JNI boundary: %s", fn, e.what());
    } catch (...) {
        LOGE("%s: unknown C++ exception at the JNI boundary", fn);
    }
    return on_throw();
}

static rnllama::llama_rn_context* to_ctx(jlong h) {
    return reinterpret_cast<rnllama::llama_rn_context*>(static_cast<uintptr_t>(h));
}

static std::string jstring_to_std(JNIEnv* env, jstring js) {
    if (js == nullptr) return "";
    const char* chars = env->GetStringUTFChars(js, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(js, chars);
    return result;
}

// A llama.cpp token can end mid-way through a multi-byte UTF-8 character
// (one visible char can span several tokens — emoji, accents, CJK).
// Passing those partial bytes to NewStringUTF aborts the VM
// ("JNI DETECTED ERROR: input is not valid Modified UTF-8").
//
// Splits buf into an emittable part (complete, valid sequences only —
// stray invalid bytes are dropped) and leaves the trailing incomplete
// sequence in buf, to be completed by the next token.
static std::string utf8_take_complete(std::string& buf) {
    const size_t n = buf.size();
    std::string out;
    out.reserve(n);
    size_t i = 0;
    size_t tail_start = n;  // start of the trailing incomplete sequence
    while (i < n) {
        const unsigned char c = buf[i];
        size_t len;
        if      (c < 0x80)           len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        else { i++; continue; }      // invalid lead byte — drop it
        if (i + len > n) { tail_start = i; break; }  // incomplete tail — hold back
        bool ok = true;
        for (size_t k = 1; k < len; k++) {
            if ((static_cast<unsigned char>(buf[i + k]) & 0xC0) != 0x80) { ok = false; break; }
        }
        if (!ok) { i++; continue; }  // invalid continuation — drop lead byte, rescan
        out.append(buf, i, len);
        i += len;
    }
    buf.erase(0, tail_start);
    return out;
}

// Non-destructive variant of utf8_take_complete(), for text that is *not* a
// stream tail we own: returns a copy stripped of any trailing incomplete
// sequence (and of stray invalid bytes). Needed because the chat parser hands
// back slices of the raw generated text, which during a partial parse can end
// mid multi-byte character — NewStringUTF() aborts the VM on those.
static std::string utf8_sanitized_copy(const std::string& s) {
    std::string buf = s;
    return utf8_take_complete(buf);
}

// ---------------------------------------------------------------------------
// Chat parse state (TEN-47)
// ---------------------------------------------------------------------------
// llama.cpp's chat layer already knows every reasoning convention there is
// (<think>…</think>, gpt-oss/harmony <|channel|>…, Gemma…): applying a chat
// template with common_chat_templates_apply() returns, besides the prompt, the
// detected format plus a serialized PEG parser able to split the model's reply
// into content / reasoning_content / tool_calls. rn-completion's
// parseChatOutput() wraps that, but only works if beginCompletion() was handed
// those three values first.
//
// nativeFormatChat() (which is where the template is applied) and
// nativeCompletion() (which is where they're needed) are two separate JNI
// calls, so the values have to survive in between. They can't live on the
// completion object — rewind()/beginCompletion() own those fields — so they're
// parked here, keyed by native context, and dropped in nativeFree().
struct chat_parse_state {
    // false when the prompt came from the legacy (non-jinja) fallback: no
    // common_chat_params, hence no format and no parser. Parsing is then
    // skipped entirely rather than run against a do-nothing parser, so the
    // caller can tell "the engine has nothing to say about this reply" from
    // "the engine says the reply is all content".
    bool                    parsable = false;
    int                     format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE;
    // Assistant prefix the template appended. common_chat_peg_parse() expects
    // it prepended to the model output, and the generated parser matches it.
    std::string             generation_prompt;
    // Serialized PEG arena (common_chat_params::parser). Empty for the
    // non-PEG formats, where common_chat_parse() falls back to a pure-content
    // parser on its own.
    std::string             parser;
    // Reasoning delimiters the same template resolution produced (TEN-85):
    // "<think>"/"</think>" for Qwen3, "<|channel>thought"/"<channel|>" for
    // Gemma-4, and so on. Both empty when the template has no notion of
    // reasoning, or when thinking was turned off for this prompt. Kept here
    // for the same reason as everything else in this struct: the template is
    // applied in nativeFormatChat() and they are needed in nativeCompletion().
    std::string             thinking_start_tag;
    std::string             thinking_end_tag;
    // The prompt this state describes. nativeCompletion() is free to be called
    // with any string (see the raw `decode()` path in LlamaEngineFlow), so the
    // state is only trusted when the prompt matches the one it came from —
    // parsing an arbitrary completion with a chat template's parser would at
    // best produce nonsense.
    std::string             prompt;
};

static std::mutex                                   g_chat_state_mu;
static std::map<const void*, chat_parse_state>      g_chat_state;

static void chat_state_put(const void* key, chat_parse_state state) {
    std::lock_guard<std::mutex> lock(g_chat_state_mu);
    g_chat_state[key] = std::move(state);
}

static chat_parse_state chat_state_get(const void* key) {
    std::lock_guard<std::mutex> lock(g_chat_state_mu);
    auto it = g_chat_state.find(key);
    return it == g_chat_state.end() ? chat_parse_state{} : it->second;
}

static void chat_state_erase(const void* key) {
    std::lock_guard<std::mutex> lock(g_chat_state_mu);
    g_chat_state.erase(key);
}

// No-op when no state is recorded for `key` (legacy template path).
static void chat_state_set_prompt(const void* key, std::string prompt) {
    std::lock_guard<std::mutex> lock(g_chat_state_mu);
    auto it = g_chat_state.find(key);
    if (it != g_chat_state.end()) it->second.prompt = std::move(prompt);
}

// ---------------------------------------------------------------------------
// nativeListBackends  (static — no model required)
// ---------------------------------------------------------------------------
// Returns a JSON array describing all available ggml backend devices.
// Uses the existing backend_devices_info() from rn-common.hpp which
// enumerates lm_ggml_backend_dev_count() devices and collects:
//   backend  — backend registry name (e.g. "Vulkan", "CPU", "OpenCL")
//   type     — "cpu" | "gpu" | "igpu" | "accel"
//   deviceName — human-readable device name (e.g. "Mali-G68 MC4")

// backend_devices_info() builds and dumps a nlohmann::json out of driver-
// supplied device names, so it throws on the same UTF-8 rule as everything
// else — the fallback is an empty JSON array, which availableBackends()
// parses into an empty list rather than a null it doesn't expect.
static jstring listBackends_impl(JNIEnv* env)
{
    std::string info = rnllama::backend_devices_info();
    LOGI("nativeListBackends: %s", info.c_str());
    return env->NewStringUTF(info.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeListBackends(
        JNIEnv* env,
        jobject /* thiz */)
{
    return jni_guard("nativeListBackends",
                     [&] { return listBackends_impl(env); },
                     [&] { return env->NewStringUTF("[]"); });
}

// ---------------------------------------------------------------------------
// nativeActiveBackend  (requires loaded model handle)
// ---------------------------------------------------------------------------
// Determines which backend is actually being used for inference.
// Approach: if n_gpu_layers > 0, find the first GPU/iGPU device in the
// ggml device registry — that device is what llama.cpp selected for
// offloading. If n_gpu_layers == 0, it's CPU.
// This is a deduction rather than a direct query because llama.cpp does
// not expose a "which backend is my context using" API at the public level.
// Returns a string like "Vulkan: Mali-G68 MC4" or "CPU: ARM CPU".

static jstring activeBackend_impl(JNIEnv* env, jlong h)
{
    if (h == 0L) {
        return env->NewStringUTF("CPU (no model loaded)");
    }
    auto* rnctx = to_ctx(h);

    const bool gpu_offload = (rnctx->params.n_gpu_layers != 0);

    if (!gpu_offload) {
        // Find CPU device name
        const size_t dev_count = lm_ggml_backend_dev_count();
        for (size_t i = 0; i < dev_count; i++) {
            lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_get(i);
            if (dev == nullptr) continue;
            if (lm_ggml_backend_dev_type(dev) == LM_GGML_BACKEND_DEVICE_TYPE_CPU) {
                const char* name = lm_ggml_backend_dev_name(dev);
                std::string result = std::string("CPU: ") + (name ? name : "ARM CPU");
                LOGI("nativeActiveBackend: %s", result.c_str());
                return env->NewStringUTF(result.c_str());
            }
        }
        return env->NewStringUTF("CPU");
    }

    // GPU offload — find first GPU or iGPU device
    const size_t dev_count = lm_ggml_backend_dev_count();
    for (size_t i = 0; i < dev_count; i++) {
        lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_get(i);
        if (dev == nullptr) continue;
        enum lm_ggml_backend_dev_type dtype = lm_ggml_backend_dev_type(dev);
        if (dtype == LM_GGML_BACKEND_DEVICE_TYPE_GPU ||
            dtype == LM_GGML_BACKEND_DEVICE_TYPE_IGPU) {
            lm_ggml_backend_reg_t reg = lm_ggml_backend_dev_backend_reg(dev);
            const char* backend_name = reg ? lm_ggml_backend_reg_name(reg) : "GPU";
            const char* dev_name = lm_ggml_backend_dev_name(dev);
            std::string result = std::string(backend_name ? backend_name : "GPU")
                               + ": " + (dev_name ? dev_name : "unknown");
            LOGI("nativeActiveBackend: %s", result.c_str());
            return env->NewStringUTF(result.c_str());
        }
    }

    // GPU requested but no GPU device found — fell back to CPU
    LOGI("nativeActiveBackend: GPU requested but no GPU device in registry, likely CPU fallback");
    return env->NewStringUTF("CPU (GPU requested but unavailable)");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeActiveBackend(
        JNIEnv* env,
        jobject /* thiz */,
        jlong h)
{
    return jni_guard("nativeActiveBackend",
                     [&] { return activeBackend_impl(env, h); },
                     [&] { return env->NewStringUTF("unknown"); });
}

// ---------------------------------------------------------------------------
// nativeReadGgufMetadata  (static — reads the GGUF header only, no weights)
// ---------------------------------------------------------------------------
// Returns a JSON string with architecture, name, file_type, context_length,
// embedding_length, block_count, param_count, vocab_size, file_size_bytes,
// plus the attention/KV-cache shape (head_count, head_count_kv,
// kv_full_elements_per_token, kv_swa_elements_per_token, sliding_window) —
// see the "KV-cache shape" block below for how those last three are derived.
// no_alloc=true → only the key-value header and tensor infos are read; a
// multi-GB model is inspected in milliseconds. Returns nullptr on failure.

static jstring readGgufMetadata_impl(JNIEnv* env, jstring path)
{
    const std::string fpath = jstring_to_std(env, path);

    lm_gguf_init_params ip{};
    ip.no_alloc = true;
    ip.ctx = nullptr;

    lm_gguf_context* gctx = lm_gguf_init_from_file(fpath.c_str(), ip);
    if (gctx == nullptr) {
        LOGE("nativeReadGgufMetadata: failed to open %s", fpath.c_str());
        return nullptr;
    }
    // Everything between here and lm_gguf_free() can throw (std::string and
    // nlohmann::json both allocate), and jni_guard() around this function
    // catches but cannot free a raw pointer it never saw. The context is owned
    // for the rest of the body so an escape on any of those paths is a caught
    // failure rather than a leak.
    const std::unique_ptr<lm_gguf_context, decltype(&lm_gguf_free)>
        gctx_owner(gctx, &lm_gguf_free);

    auto get_str = [&](const std::string& key) -> std::string {
        int64_t i = lm_gguf_find_key(gctx, key.c_str());
        return (i >= 0 && lm_gguf_get_kv_type(gctx, i) == LM_GGUF_TYPE_STRING)
             ? lm_gguf_get_val_str(gctx, i) : "";
    };
    // Integer keys vary between u32/u64 across models — handle both.
    auto get_uint = [&](const std::string& key) -> uint64_t {
        int64_t i = lm_gguf_find_key(gctx, key.c_str());
        if (i < 0) return 0;
        switch (lm_gguf_get_kv_type(gctx, i)) {
            case LM_GGUF_TYPE_UINT32: return lm_gguf_get_val_u32(gctx, i);
            case LM_GGUF_TYPE_UINT64: return lm_gguf_get_val_u64(gctx, i);
            default: return 0;
        }
    };

    // Some attention keys are per-layer arrays rather than scalars (Gemma 4's
    // attention.head_count_kv is [8, 8, 8, 8, 8, 2, …] — see llama-model.cpp's
    // get_key_or_arr for head_count/head_count_kv). Fills all `n` entries of
    // `out`: a scalar is broadcast, an array of exactly `n` entries is copied.
    // Returns false when the key is missing, is an array of some other length,
    // or holds anything that isn't a plain unsigned count — callers must then
    // keep their own default rather than act on a value that isn't the one
    // llama.cpp would use, and must not read `out`, which a rejected array may
    // have partially written.
    auto get_uint_per_layer = [&](const std::string& key, std::vector<uint64_t>& out, size_t n) -> bool {
        int64_t i = lm_gguf_find_key(gctx, key.c_str());
        if (i < 0) return false;

        const enum lm_gguf_type kv_type = lm_gguf_get_kv_type(gctx, i);
        if (kv_type != LM_GGUF_TYPE_ARRAY) {
            const uint64_t scalar = get_uint(key);
            if (scalar == 0) return false;
            std::fill(out.begin(), out.begin() + n, scalar);
            return true;
        }

        if (lm_gguf_get_arr_n(gctx, i) != n) return false;

        const void* data = lm_gguf_get_arr_data(gctx, i);
        switch (lm_gguf_get_arr_type(gctx, i)) {
            case LM_GGUF_TYPE_UINT32:
                for (size_t il = 0; il < n; il++) out[il] = static_cast<const uint32_t*>(data)[il];
                return true;
            case LM_GGUF_TYPE_INT32:
                for (size_t il = 0; il < n; il++) {
                    const int32_t v = static_cast<const int32_t*>(data)[il];
                    if (v < 0) return false;
                    out[il] = static_cast<uint64_t>(v);
                }
                return true;
            default:
                return false;
        }
    };

    // Bool arrays are stored as int8 (see gguf.h). Same all-or-nothing contract
    // as get_uint_per_layer: a length mismatch or any other type reads as "not
    // available" rather than as a partial answer.
    auto get_bool_per_layer = [&](const std::string& key, std::vector<uint8_t>& out, size_t n) -> bool {
        int64_t i = lm_gguf_find_key(gctx, key.c_str());
        if (i < 0) return false;
        if (lm_gguf_get_kv_type(gctx, i) != LM_GGUF_TYPE_ARRAY) return false;
        if (lm_gguf_get_arr_type(gctx, i) != LM_GGUF_TYPE_BOOL) return false;
        if (lm_gguf_get_arr_n(gctx, i) != n) return false;

        const auto* data = static_cast<const int8_t*>(lm_gguf_get_arr_data(gctx, i));
        for (size_t il = 0; il < n; il++) out[il] = data[il] != 0 ? 1 : 0;
        return true;
    };

    const std::string arch = get_str("general.architecture");

    // Parameter count: sum of elements over all tensor infos
    uint64_t n_params = 0;
    const int64_t n_tensors = lm_gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n_tensors; i++) {
        const enum lm_ggml_type t = lm_gguf_get_tensor_type(gctx, i);
        const size_t sz = lm_gguf_get_tensor_size(gctx, i);
        n_params += sz / lm_ggml_type_size(t) * lm_ggml_blck_size(t);
    }

    // Vocab size = length of the tokenizer tokens array
    uint64_t vocab = 0;
    {
        int64_t i = lm_gguf_find_key(gctx, "tokenizer.ggml.tokens");
        if (i >= 0) vocab = lm_gguf_get_arr_n(gctx, i);
    }

    // Thinking capability: no dedicated GGUF key — detect from the chat
    // template (jinja string in the header). Thinking-capable templates
    // reference enable_thinking (Qwen3) or emit <think> blocks.
    bool supports_thinking = false;
    {
        const std::string tmpl = get_str("tokenizer.chat_template");
        supports_thinking = tmpl.find("enable_thinking") != std::string::npos
                         || tmpl.find("<think>") != std::string::npos;
    }

    const uint64_t n_layer = get_uint(arch + ".block_count");
    const uint64_t n_embd  = get_uint(arch + ".embedding_length");

    // -----------------------------------------------------------------
    // KV-cache shape
    //
    // llama_kv_cache allocates, per layer that owns a cache, a K tensor of
    // n_embd_k_gqa(il) * kv_size elements and a V tensor of n_embd_v_gqa(il) *
    // kv_size, where n_embd_k_gqa(il) = n_embd_head_k(il) * n_head_kv(il)
    // (llama-hparams.cpp). Summing that over the layers gives the elements per
    // context token — the number a caller needs to size a load, and one that
    // n_embd * n_layer only matches for plain multi-head attention.
    //
    // The V half looks like an exception and isn't: at [TAG_V_CACHE_VARIABLE]
    // in llama-kv-cache.cpp a layer's V tensor is sized at n_embd_v_gqa_max(),
    // the largest any layer needs, rather than at its own n_embd_v_gqa(il)
    // whenever v_trans, and v_trans = !cparams.flash_attn (llama-model.cpp).
    // But cparams.flash_attn is still just `flash_attn_type != DISABLED` at
    // that point: llama_context's constructor calls create_memory() well
    // before the `if (cparams.auto_fa)` block that turns AUTO into a real
    // answer, and never rebuilds the cache afterwards. So AUTO and ENABLED
    // both build it with v_trans = false, and only an explicit "off" gets the
    // padding. The load path leaves the type on AUTO unless a caller says
    // otherwise, and an "off" caller cannot ask for a quantised V cache at all
    // (llama-context.cpp rejects that pair), so the V half is charged per
    // layer. That under-counts a load that explicitly disables flash
    // attention, by the spread between the largest V row and the smaller ones
    // — nothing at all on a model with uniform head counts, and bounded by the
    // windowed half on a Gemma, whose windowed cache is a few hundred cells.
    //
    // Two buckets, because llama_kv_cache_iswa allocates two caches: the
    // full-attention layers at kv_size = n_ctx, and the sliding-window layers
    // at kv_size = pad256(min(n_ctx, sliding_window + one ubatch)) instead.
    // Both kv_size values are whole multiples of 256 cells — llama_context
    // pads n_ctx itself before any cache is built — so a caller multiplying
    // these counts by a raw n_ctx under-estimates unless it pads first.
    // Callers that don't care about the second bucket can just add the two and
    // treat both as growing with n_ctx — that over-estimates, which is the
    // safe direction.
    //
    // Every read here fails closed: a key that is missing, the wrong type or
    // the wrong length leaves the layer in the full-attention bucket at the
    // larger of its two head dimensions, and a model whose head counts can't
    // be read at all reports 0 so the caller knows to fall back to its own
    // estimate. Over-estimating a KV cache costs context; under-estimating it
    // costs an out-of-memory kill at load time.
    // -----------------------------------------------------------------
    uint64_t kv_full_elements = 0;  // elements per token in layers sized by n_ctx
    uint64_t kv_swa_elements  = 0;  // elements per token in sliding-window layers
    uint64_t sliding_window   = 0;  // 0 unless kv_swa_elements > 0
    uint64_t head_count       = 0;
    uint64_t head_count_kv    = 0;

    if (n_layer > 0 && n_embd > 0) {
        std::vector<uint64_t> n_head   (n_layer, 0);
        std::vector<uint64_t> n_head_kv(n_layer, 0);

        const bool have_head    = get_uint_per_layer(arch + ".attention.head_count",    n_head,    n_layer);
        const bool have_head_kv = get_uint_per_layer(arch + ".attention.head_count_kv", n_head_kv, n_layer);

        if (have_head && have_head_kv) {
            head_count    = *std::max_element(n_head.begin(),    n_head.end());
            head_count_kv = *std::max_element(n_head_kv.begin(), n_head_kv.end());

            // llama.cpp defaults the head dimensions to n_embd / n_head() —
            // n_head() being layer 0's head count — and overrides them with
            // attention.key_length / .value_length when those are present.
            const uint64_t head_dim_default = n_head[0] > 0 ? n_embd / n_head[0] : 0;

            uint64_t k_full = get_uint(arch + ".attention.key_length");
            uint64_t v_full = get_uint(arch + ".attention.value_length");
            if (k_full == 0) k_full = head_dim_default;
            if (v_full == 0) v_full = head_dim_default;

            uint64_t k_swa = get_uint(arch + ".attention.key_length_swa");
            uint64_t v_swa = get_uint(arch + ".attention.value_length_swa");
            if (k_swa == 0) k_swa = k_full;
            if (v_swa == 0) v_swa = v_full;

            // Layers from this index on reuse an earlier layer's cache rather
            // than owning one: llama_kv_cache's constructor skips every layer
            // for which hparams.has_kv(il) is false before it allocates
            // anything, and the layer_reuse_cb in llama-model.cpp then points
            // those layers at a cache that already exists. has_kv(il) is
            // `il < n_layer_kv_from_start` (llama-hparams.cpp), so the
            // cache-owning layers are always a prefix.
            //
            // Only the architectures whose loader in this tree sets
            // n_layer_kv_from_start have one: gemma4 derives it from
            // attention.shared_kv_layers (models/gemma4.cpp) and gemma3n
            // hardcodes twenty regardless of what the file says
            // (models/gemma3n.cpp). Every other architecture leaves it at -1
            // and keeps all n_layer caches, even if it happens to carry the
            // key.
            uint64_t n_layer_kv = n_layer;
            if (arch == "gemma4" || arch == "gemma4-assistant") {
                const uint64_t shared = get_uint(arch + ".attention.shared_kv_layers");
                if (shared < n_layer) n_layer_kv = n_layer - shared;
            } else if (arch == "gemma3n") {
                n_layer_kv = std::min<uint64_t>(20, n_layer);
            }

            // Which layers are windowed, taken only from an explicit per-layer
            // bool array of exactly n_layer entries, and only for the
            // architectures that feed that array straight into
            // hparams.is_swa_impl (models/gemma4.cpp again). The key's scalar
            // form is a *period* whose phase varies per architecture
            // (llama_hparams::set_swa_pattern's dense_first), so it is
            // deliberately not interpreted: reading it wrong would mark a
            // full-attention layer as windowed and under-estimate the cache.
            std::vector<uint8_t> is_swa(n_layer, 0);
            const uint64_t n_swa = get_uint(arch + ".attention.sliding_window");
            const bool have_swa = n_swa > 0
                && (arch == "gemma4" || arch == "gemma4-assistant")
                && get_bool_per_layer(arch + ".attention.sliding_window_pattern", is_swa, n_layer);

            if (!have_swa) {
                // No idea which layers are windowed, so every layer is charged
                // at whichever of the two head dimensions is larger.
                k_full = std::max(k_full, k_swa);
                v_full = std::max(v_full, v_swa);
            }

            // n_embd_k_gqa(il) + n_embd_v_gqa(il), summed over the layers that
            // own a cache and split by the bucket llama_kv_cache_iswa's two
            // filters would put each one in (llama-kv-cache-iswa.cpp: the base
            // cache takes !is_swa(il), the SWA cache takes is_swa(il)).
            for (uint64_t il = 0; il < n_layer_kv; il++) {
                const bool swa = have_swa && is_swa[il];
                const uint64_t elements = n_head_kv[il]
                    * ((swa ? k_swa : k_full) + (swa ? v_swa : v_full));

                if (swa) {
                    kv_swa_elements += elements;
                } else {
                    kv_full_elements += elements;
                }
            }

            if (kv_swa_elements > 0) sliding_window = n_swa;
        }
    }

    struct stat st{};
    const uint64_t file_size = (stat(fpath.c_str(), &st) == 0)
        ? static_cast<uint64_t>(st.st_size) : 0;

    nlohmann::json j = {
        {"architecture",     arch},
        {"name",             get_str("general.name")},
        {"file_type",        get_uint("general.file_type")},
        {"context_length",   get_uint(arch + ".context_length")},
        {"embedding_length", n_embd},
        {"block_count",      n_layer},
        {"param_count",      n_params},
        {"vocab_size",       vocab},
        {"file_size_bytes",  file_size},
        {"supports_thinking", supports_thinking},
        {"head_count",              head_count},
        {"head_count_kv",           head_count_kv},
        {"kv_full_elements_per_token", kv_full_elements},
        {"kv_swa_elements_per_token",  kv_swa_elements},
        {"sliding_window",             sliding_window},
        {"kv_shape_version",           TENSAI_KV_SHAPE_VERSION},
    };

    // `replace`, not the default `strict` handler. `architecture` and `name`
    // come raw out of the GGUF header, which is a length followed by that many
    // bytes with nothing checking them (gguf.cpp): a truncated download or a
    // file that was never a GGUF puts arbitrary bytes in a std::string here.
    // The strict handler answers that with type_error.316, which under the old
    // code left this function as a C++ exception and killed the process.
    //
    // Replacing each bad byte with U+FFFD keeps a model whose name is merely
    // mis-encoded importable — the name renders with a few replacement
    // characters instead of the whole file being refused — and U+FFFD is
    // ordinary 3-byte UTF-8, so NewStringUTF() below takes it as-is.
    const std::string out = j.dump(-1, ' ', false,
                                   nlohmann::json::error_handler_t::replace);
    LOGI("nativeReadGgufMetadata: %s", out.c_str());
    return env->NewStringUTF(out.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeReadGgufMetadata(
        JNIEnv* env,
        jclass /* clazz */,
        jstring path)
{
    // nullptr is this function's documented "not a readable GGUF" answer, and
    // both callers already handle it: readMetadata() turns it into a
    // ModelLoadException the import screen reports, and KvCacheShapeBackfill
    // skips the row.
    return jni_guard("nativeReadGgufMetadata",
                     [&] { return readGgufMetadata_impl(env, path); },
                     [] { return (jstring) nullptr; });
}

// ---------------------------------------------------------------------------
// nativeKvShapeVersion  (static — no model, no file)
// ---------------------------------------------------------------------------
// The same TENSAI_KV_SHAPE_VERSION that nativeReadGgufMetadata stamps into
// every result, answerable without a GGUF to read. A caller holding persisted
// KV shapes needs it to decide *which* of them are stale before it opens
// anything — see the constant's comment at the top of this file.

extern "C" JNIEXPORT jlong JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeKvShapeVersion(
        JNIEnv* /* env */,
        jclass /* clazz */)
{
    return static_cast<jlong>(TENSAI_KV_SHAPE_VERSION);
}

// ---------------------------------------------------------------------------
// nativeLoadModel
// ---------------------------------------------------------------------------

// Model-loading progress trampoline. The callback fires synchronously on the
// thread executing nativeLoadModel, so the JNIEnv* and local refs in the
// holder stay valid for the whole load. Returning false aborts the load.
struct LoadProgressHolder {
    JNIEnv*   env;
    jobject   cb;
    jmethodID onProgress;
};

static bool load_progress_trampoline(float progress, void* user_data) {
    auto* h = static_cast<LoadProgressHolder*>(user_data);
    jboolean keep_going = h->env->CallBooleanMethod(h->cb, h->onProgress, (jfloat) progress);
    if (h->env->ExceptionCheck()) {
        h->env->ExceptionClear();
        LOGE("load progress callback threw — aborting load");
        return false;
    }
    return keep_going == JNI_TRUE;
}

static jlong loadModel_impl(
        JNIEnv* env,
        jstring path,
        jint nGpuLayers,
        jint nCtx,
        jint nThreads,
        jobject progressCb,
        jstring kvCacheType,
        jstring flashAttn)
{
    // Owned rather than raw: loadModel() below runs llama.cpp's whole model
    // loader, which throws on a file it can't make sense of, and the context
    // would otherwise leak past jni_guard() — a leak measured in the model's
    // own footprint, not in bytes.
    auto rnctx_owner = std::make_unique<rnllama::llama_rn_context>();
    auto* rnctx = rnctx_owner.get();

    common_params p;
    p.model.path   = jstring_to_std(env, path);
    p.n_gpu_layers = static_cast<int32_t>(nGpuLayers);
    p.n_ctx        = static_cast<int32_t>(nCtx > 0 ? nCtx : 4096);
    if (p.n_batch <= 0) p.n_batch = 512;

    // nThreads <= 0 → keep llama.cpp auto-detect. Explicit value pins both
    // the generation and batch (prefill) thread pools — on big.LITTLE SoCs
    // fewer threads on big cores can beat auto across all cores.
    if (nThreads > 0) {
        p.cpuparams.n_threads       = static_cast<int32_t>(nThreads);
        p.cpuparams_batch.n_threads = static_cast<int32_t>(nThreads);
    }

    // KV cache quantization — empty/invalid falls back to f16 inside
    // kv_cache_type_from_str. Applied to both K and V.
    const std::string kv_type = jstring_to_std(env, kvCacheType);
    if (!kv_type.empty()) {
        p.cache_type_k = rnllama::kv_cache_type_from_str(kv_type);
        p.cache_type_v = rnllama::kv_cache_type_from_str(kv_type);
    }

    // Flash attention: "on"/"off"; anything else keeps AUTO
    const std::string fa = jstring_to_std(env, flashAttn);
    if (!fa.empty()) {
        p.flash_attn_type = rnllama::flash_attn_type_from_str(fa);
    }

    LOGI("loadModel: path=%s n_gpu_layers=%d n_ctx=%d n_batch=%d n_threads=%d progress_cb=%d kv_cache=%s",
         p.model.path.c_str(), p.n_gpu_layers, p.n_ctx, p.n_batch,
         nThreads > 0 ? nThreads : -1, progressCb != nullptr,
         kv_type.empty() ? "f16(default)" : kv_type.c_str());

    LoadProgressHolder holder{};
    jclass cbClass = nullptr;
    if (progressCb != nullptr) {
        cbClass = env->GetObjectClass(progressCb);
        holder.env = env;
        holder.cb = progressCb;
        holder.onProgress = env->GetMethodID(cbClass, "onProgress", "(F)Z");
        if (holder.onProgress != nullptr) {
            p.load_progress_callback = load_progress_trampoline;
            p.load_progress_callback_user_data = &holder;
        } else {
            LOGE("loadModel: onProgress(F)Z not found, progress disabled");
        }
    }

    const bool ok = rnctx->loadModel(p);
    if (cbClass != nullptr) env->DeleteLocalRef(cbClass);

    if (!ok) {
        LOGE("loadModel failed (or aborted by progress callback)");
        return 0L;
    }

    LOGI("loadModel OK, ptr=%p", rnctx);
    // Ownership moves to the Kotlin side, which gives it back in nativeFree().
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(rnctx_owner.release()));
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeLoadModel(
        JNIEnv* env,
        jobject /* thiz */,
        jstring path,
        jint nGpuLayers,
        jint nCtx,
        jint nThreads,
        jobject progressCb,
        jstring kvCacheType,
        jstring flashAttn)
{
    // 0 is the handle value the Kotlin side already reads as "load failed".
    return jni_guard("nativeLoadModel",
                     [&] {
                         return loadModel_impl(env, path, nGpuLayers, nCtx, nThreads,
                                               progressCb, kvCacheType, flashAttn);
                     },
                     [] { return (jlong) 0; });
}

// ---------------------------------------------------------------------------
// nativeFree
// ---------------------------------------------------------------------------

static void free_impl(jlong h)
{
    if (h == 0L) return;
    auto* rnctx = to_ctx(h);
    LOGI("nativeFree ptr=%p", rnctx);
    chat_state_erase(rnctx);
    delete rnctx;
}

extern "C" JNIEXPORT void JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeFree(
        JNIEnv* /* env */,
        jobject /* thiz */,
        jlong h)
{
    // Nothing to answer with — close() is the one call the app makes while
    // shutting an engine down, and a throw here must not be what stops it.
    jni_guard("nativeFree", [&] { free_impl(h); }, [] {});
}

// ---------------------------------------------------------------------------
// nativeCompletion
// ---------------------------------------------------------------------------
// Kotlin interface attendue : interface TokenCallback { fun onToken(token: String) }
// cbChat (nullable) : interface ChatParseCallback {
//     fun onChatParse(content: String, reasoningContent: String) }
//
// Returns the number of tokens the decode actually sampled
// (completion->num_tokens_predicted), which is the only figure that can be
// compared against n_predict to tell "the model ended its turn" from "the
// model ran out of budget". Counting onToken() calls on the Kotlin side is
// not that figure and never was (TEN-79): `emit` below holds tokens back —
// for an incomplete UTF-8 sequence (any accented character split across two
// tokens), and for a partial stop-sequence match — so a callback can cover
// several tokens, or none. Under-counting there reads as a Complete reply on
// a decode that was cut off mid-sentence. Zero on every early exit: nothing
// was generated.

static jint completion_impl(
        JNIEnv* env,
        jlong   h,
        jstring prompt,
        jint    nPredict,
        jint    reasoningBudgetTokens,
        jfloat  temperature,
        jint    topK,
        jfloat  topP,
        jfloat  minP,
        jobjectArray stopSequences,
        jobject cb,
        jobject cbChat)
{
    if (h == 0L) return 0;
    auto* rnctx = to_ctx(h);

    if (rnctx->completion == nullptr) {
        LOGE("nativeCompletion: completion context is null");
        return 0;
    }

    // Resolve callback method once
    jclass cbClass = env->GetObjectClass(cb);
    jmethodID onToken = env->GetMethodID(cbClass, "onToken", "(Ljava/lang/String;)V");
    if (onToken == nullptr) {
        LOGE("nativeCompletion: onToken method not found");
        env->DeleteLocalRef(cbClass);
        return 0;
    }

    jclass    chatCbClass = nullptr;
    jmethodID onChatParse = nullptr;
    if (cbChat != nullptr) {
        chatCbClass = env->GetObjectClass(cbChat);
        onChatParse = env->GetMethodID(chatCbClass, "onChatParse",
                                       "(Ljava/lang/String;Ljava/lang/String;)V");
        if (onChatParse == nullptr) {
            LOGE("nativeCompletion: onChatParse method not found");
            env->DeleteLocalRef(chatCbClass);
            chatCbClass = nullptr;
        }
    }

    // Set prompt, n_predict cap, and rewind completion state
    rnctx->params.prompt = jstring_to_std(env, prompt);
    rnctx->params.n_predict = static_cast<int32_t>(nPredict > 0 ? nPredict : 512);

    // Sampling parameters — read by common_sampler_init() inside initSampling().
    // temp <= 0 → greedy decoding; topK 0 / topP 1.0 / minP 0 disable those filters.
    rnctx->params.sampling.temp  = static_cast<float>(temperature);
    rnctx->params.sampling.top_k = static_cast<int32_t>(topK);
    rnctx->params.sampling.top_p = static_cast<float>(topP);
    rnctx->params.sampling.min_p = static_cast<float>(minP);
    LOGI("nativeCompletion: n_predict=%d temp=%.2f top_k=%d top_p=%.2f min_p=%.2f",
         rnctx->params.n_predict, rnctx->params.sampling.temp,
         rnctx->params.sampling.top_k, rnctx->params.sampling.top_p,
         rnctx->params.sampling.min_p);

    auto* comp = rnctx->completion;
    comp->rewind();  // clears params.antiprompt — set stop sequences after this

    // Stop sequences → antiprompt, checked via findStoppingStrings() below
    const jsize n_stops = stopSequences ? env->GetArrayLength(stopSequences) : 0;
    for (jsize i = 0; i < n_stops; i++) {
        auto js = (jstring) env->GetObjectArrayElement(stopSequences, i);
        std::string s = jstring_to_std(env, js);
        env->DeleteLocalRef(js);
        if (!s.empty()) rnctx->params.antiprompt.push_back(std::move(s));
    }
    const bool has_stops = !rnctx->params.antiprompt.empty();
    if (has_stops) {
        LOGI("nativeCompletion: %d stop sequences", (int) rnctx->params.antiprompt.size());
    }

    // TEN-47: what nativeFormatChat() resolved from the model's chat template.
    // Read here rather than after loadPrompt() (where it used to be) because
    // the reasoning budget below has to be in params *before* initSampling():
    // common_sampler_init() is what builds the budget sampler out of them, and
    // setting them afterwards would build nothing and drop the budget silently.
    const chat_parse_state chat_state = chat_state_get(rnctx);
    const bool chat_state_applies = chat_state.parsable
                                 && chat_state.prompt == rnctx->params.prompt;

    // -----------------------------------------------------------------------
    // Reasoning budget (TEN-85)
    // -----------------------------------------------------------------------
    // A cap on how many tokens the model may spend *inside* its reasoning
    // block. On exhaustion common/reasoning-budget.cpp forces the template's
    // own end tag out token by token (every other logit to -inf, waiting for a
    // multi-byte character to finish first) and then steps aside: the model
    // goes on to write its answer. That is the whole point — the caller used to
    // have to interrupt the decode instead, which left the user with reasoning
    // and no reply.
    //
    // Deliberately derived from the chat template rather than from a hardcoded
    // "<think>": the tags are whatever common_chat_templates_apply() resolved
    // for this model (see chat_parse_state), which is the same source TEN-47's
    // reply parser comes from. A model whose template knows nothing about
    // reasoning gets no sampler and generates exactly as before.
    //
    // rewind() clears antiprompt and the grammar fields but not these, and
    // params outlive the call, so a completion asking for no budget has to
    // clear what the last one left behind or it inherits its cap.
    rnctx->params.sampling.reasoning_budget_tokens = -1;
    rnctx->params.sampling.reasoning_budget_start.clear();
    rnctx->params.sampling.reasoning_budget_end.clear();
    rnctx->params.sampling.reasoning_budget_forced.clear();

    if (reasoningBudgetTokens >= 0) {
        // Both tags are needed: without an end tag there is nothing to force,
        // and common_sampler_init() skips the sampler outright without a start
        // sequence to arm on.
        if (chat_state_applies && !chat_state.thinking_start_tag.empty()
                               && !chat_state.thinking_end_tag.empty()) {
            const llama_vocab* vocab = llama_model_get_vocab(rnctx->model);
            auto& sp = rnctx->params.sampling;
            sp.reasoning_budget_tokens = static_cast<int32_t>(reasoningBudgetTokens);
            sp.reasoning_budget_start  = common_tokenize(vocab, chat_state.thinking_start_tag, false, true);
            sp.reasoning_budget_end    = common_tokenize(vocab, chat_state.thinking_end_tag, false, true);
            // No budget message prepended to the forced sequence: any text
            // injected here would land in the reasoning the user reads,
            // attributed to a model that never wrote it.
            sp.reasoning_budget_forced = sp.reasoning_budget_end;
            // The assistant prefix the template already appended to the prompt.
            // common_sampler_init() replays it through the sampler, so a
            // template that opens the reasoning block itself (the prefix ends
            // in the start tag) starts out counting rather than waiting for a
            // tag the model has no reason to emit — the state the sampler's
            // REASONING_BUDGET_COUNTING initial state exists for, reached here
            // by the general route instead of by guessing per model.
            sp.generation_prompt = chat_state.generation_prompt;
            LOGI("nativeCompletion: reasoning budget=%d tokens, start='%s' (%zu tok), end='%s' (%zu tok)",
                 sp.reasoning_budget_tokens,
                 chat_state.thinking_start_tag.c_str(), sp.reasoning_budget_start.size(),
                 chat_state.thinking_end_tag.c_str(),   sp.reasoning_budget_end.size());
        } else {
            LOGI("nativeCompletion: reasoning budget=%d ignored, this prompt has no reasoning delimiters",
                 (int) reasoningBudgetTokens);
        }
    }

    if (!comp->initSampling()) {
        LOGE("nativeCompletion: initSampling failed");
        env->DeleteLocalRef(cbClass);
        if (chatCbClass != nullptr) env->DeleteLocalRef(chatCbClass);
        return 0;
    }

    // Tokenise and load prompt (no media)
    comp->loadPrompt({});

    // TEN-47: hand the chat format / reasoning format / generation prompt /
    // PEG parser that nativeFormatChat() detected to the completion, which is
    // what makes parseChatOutput() below able to split reasoning from content.
    // Without them beginCompletion() defaults to CONTENT_ONLY + reasoning NONE
    // and every reasoning marker the model emits stays in the content.
    // (chat_state / chat_state_applies are resolved above, before sampling.)
    const bool parse_chat = chat_state_applies && onChatParse != nullptr;

    if (chat_state_applies) {
        comp->beginCompletion(chat_state.format, chat_state.reasoning_format,
                              chat_state.generation_prompt, chat_state.parser);
    } else {
        // Raw (non-chat) completion, or a prompt this context never formatted:
        // the pre-TEN-47 behaviour, everything is content.
        comp->beginCompletion();
    }

    // Cumulative content / reasoning last handed to onChatParse — the callback
    // is skipped when a token changes neither (the parser holds text back
    // while a marker is still ambiguous), so the Kotlin side never sees a
    // no-op update and never allocates a String for one.
    std::string last_content;
    std::string last_reasoning;
    bool parse_failed_logged = false;

    // Re-parses the whole reply so far. Deliberately: llama.cpp's chat parser
    // is not incremental, and a partial parse can *retract* text (bytes that
    // read as content become part of a marker once the next token lands), so
    // there is no correct way to emit deltas from it without reimplementing
    // the reconciliation. Cost measured off-device on the parser shape a
    // tag-based reasoning template generates: ~6 µs per KB of accumulated
    // reply, ~56 ms in total for a whole 2048-token generation parsed on every
    // single token (x86; call it 3-5× that on a mid-range Cortex-A78, so a few
    // hundred ms) — against a generation that takes minutes on the same core.
    // Well under a percent of wall time, so no throttling: the alternative
    // (parse every N tokens) would trade an always-correct stream for a
    // saving that does not exist.
    auto emit_parse = [&](bool is_partial) {
        if (!parse_chat) return;
        rnllama::completion_chat_output out;
        try {
            out = comp->parseChatOutput(is_partial);
        } catch (const std::exception& e) {
            // common_chat_parse() throws when the reply does not match the
            // format the template advertised. Not fatal — the raw token
            // stream is unaffected, the caller just gets no split for this
            // reply and falls back to whatever it does without one.
            if (!parse_failed_logged) {
                LOGE("nativeCompletion: chat parse failed (%s) — reasoning split unavailable", e.what());
                parse_failed_logged = true;
            }
            return;
        }
        if (out.content.empty() && out.reasoning_content.empty()) return;
        if (out.content == last_content && out.reasoning_content == last_reasoning) return;
        last_content   = out.content;
        last_reasoning = out.reasoning_content;

        // Both can end mid multi-byte character on a partial parse.
        const std::string safe_content   = utf8_sanitized_copy(last_content);
        const std::string safe_reasoning = utf8_sanitized_copy(last_reasoning);
        jstring jcontent   = env->NewStringUTF(safe_content.c_str());
        jstring jreasoning = env->NewStringUTF(safe_reasoning.c_str());
        if (jcontent != nullptr && jreasoning != nullptr) {
            env->CallVoidMethod(cbChat, onChatParse, jcontent, jreasoning);
        }
        if (jcontent   != nullptr) env->DeleteLocalRef(jcontent);
        if (jreasoning != nullptr) env->DeleteLocalRef(jreasoning);
    };

    // UTF-8 holdback buffer: only complete sequences ever reach NewStringUTF.
    std::string utf8_buf;
    auto emit = [&](const std::string& s) {
        if (s.empty()) return;
        utf8_buf += s;
        const std::string ready = utf8_take_complete(utf8_buf);
        if (ready.empty()) return;
        jstring jtok = env->NewStringUTF(ready.c_str());
        env->CallVoidMethod(cb, onToken, jtok);
        env->DeleteLocalRef(jtok);
    };

    // With stop sequences, tokens are staged in `pending` and only emitted
    // once they can no longer be part of a stop word (a stop can span
    // several tokens). On a full match, everything from the match start
    // is dropped so the stream never contains the stop sequence.
    std::string pending;

    // Offset in generated_text where a matched stop word starts, so the final
    // parse can be run on the reply without it (findStoppingStrings() does not
    // truncate generated_text itself).
    size_t stop_trim_pos = std::string::npos;

    while (comp->has_next_token && !comp->is_interrupted) {
        rnllama::completion_token_output tok_out = comp->doCompletion();

        if (tok_out.tok == -1) {
            break;
        }

        const std::string& text = tok_out.text;

        if (!has_stops) {
            emit(text);
            emit_parse(/* is_partial */ true);
            continue;
        }

        pending += text;

        // Full stop word in the generated text? (doCompletion already
        // appended `text` to comp->generated_text.)
        size_t stop_pos = comp->findStoppingStrings(comp->generated_text, text.size(), rnllama::STOP_FULL);
        if (comp->stopped_word) {
            const size_t tail_after_stop = comp->generated_text.size() - stop_pos;
            if (pending.size() >= tail_after_stop) {
                emit(pending.substr(0, pending.size() - tail_after_stop));
            }
            pending.clear();
            stop_trim_pos = stop_pos;
            break;  // has_next_token was set false by findStoppingStrings
        }

        // Partial stop match at the tail: hold back only the matching suffix
        size_t partial_pos = comp->findStoppingStrings(pending, 0, rnllama::STOP_PARTIAL);
        if (partial_pos == std::string::npos) {
            emit(pending);
            pending.clear();
        } else if (partial_pos > 0) {
            emit(pending.substr(0, partial_pos));
            pending.erase(0, partial_pos);
        }
        emit_parse(/* is_partial */ true);
    }

    // Generation ended without a stop match (EOS / n_predict / interrupt):
    // flush whatever was held back. Anything still in utf8_buf after this
    // is an incomplete sequence that can never render — dropped by design.
    if (!pending.empty() && !comp->is_interrupted) {
        emit(pending);
    }

    // Final, non-partial parse: the partial parser is lenient (it has to
    // tolerate a marker arriving half-written) where the full one resolves the
    // reply for good. Dropping the stop word first, so it doesn't land in
    // `content` — the token stream never contained it either.
    if (parse_chat) {
        if (stop_trim_pos != std::string::npos && stop_trim_pos <= comp->generated_text.size()) {
            comp->generated_text.erase(stop_trim_pos);
        }
        // An interrupted reply is by definition unfinished — parsing it as
        // complete would only throw on a marker the model never got to close.
        emit_parse(/* is_partial */ comp->is_interrupted);
    }

    env->DeleteLocalRef(cbClass);
    if (chatCbClass != nullptr) env->DeleteLocalRef(chatCbClass);

    // Read after the loop, not accumulated alongside it: rewind() zeroes it on
    // entry and doCompletion() is the only thing that raises it, so this is the
    // decode's own count and nothing else's.
    return static_cast<jint>(comp->num_tokens_predicted);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeCompletion(
        JNIEnv* env,
        jobject /* thiz */,
        jlong   h,
        jstring prompt,
        jint    nPredict,
        jint    reasoningBudgetTokens,
        jfloat  temperature,
        jint    topK,
        jfloat  topP,
        jfloat  minP,
        jobjectArray stopSequences,
        jobject cb,
        jobject cbChat)
{
    // -1, not the 0 the early exits above use. 0 is a truthful count — nothing
    // was sampled — and the Kotlin side reads counts by comparing them against
    // n_predict: `tokensPredicted < nPredict` is StopReason.Complete, so a
    // decode that died on a C++ exception would be told to the user as a reply
    // the model finished on its own. That is a quieter lie than the crash this
    // guard replaced, and this project has spent three tickets removing exactly
    // that kind of lie.
    //
    // A negative count is not a count at all, so it cannot be mistaken for one.
    // LlamaKtEngineImpl.decode() turns it into a DecodeEvent.Error, the same
    // terminal event an OOM or a Kotlin-side exception produces.
    return jni_guard("nativeCompletion",
                     [&] {
                         return completion_impl(env, h, prompt, nPredict,
                                                reasoningBudgetTokens, temperature,
                                                topK, topP, minP, stopSequences,
                                                cb, cbChat);
                     },
                     [] { return (jint) -1; });
}

// ---------------------------------------------------------------------------
// nativeFormatChat
// ---------------------------------------------------------------------------
// Applies the model's built-in chat template (from GGUF metadata) to a JSON
// array of messages: [{"role":"user","content":"..."},...]
// Passing an empty chat_template string uses the template embedded in the GGUF.
// Returns the fully formatted prompt string ready for completion.

static jstring formatChat_impl(
        JNIEnv* env,
        jlong   h,
        jstring messagesJson,
        jboolean enableThinking)
{
    if (h == 0L) return env->NewStringUTF("");
    auto* rnctx = to_ctx(h);

    std::string msgs = jstring_to_std(env, messagesJson);

    std::string formatted;
    try {
        // Jinja path: honors enable_thinking (Qwen3 & other thinking models
        // render an empty think block / no think prompt when disabled).
        // Empty chat_template → template stored in the GGUF (rnctx->templates).
        // add_generation_prompt = true appends the assistant prefix.
        //
        // reasoning_format is "auto", not "none": it is not just a post-hoc
        // display setting, it is baked into the parser generated here.
        // common_chat_templates_apply() → autoparser::build_parser() only wires
        // a reasoning rule when reasoning_format != NONE (see
        // chat-auto-parser-generator.cpp), so asking for "none" hands back a
        // parser that can only ever report everything — reasoning markers
        // included — as content. That was TEN-47: Gemma/gpt-oss channel
        // markers rendered raw in the chat.
        const common_chat_params params = rnctx->getFormattedChatWithJinja(
            msgs,
            /* chat_template     */ "",
            /* json_schema       */ "",
            /* tools             */ "",
            /* parallel_tool_call*/ false,
            /* tool_choice       */ "",
            /* enable_thinking   */ enableThinking == JNI_TRUE,
            /* reasoning_format  */ "auto",  // "" throws (Unknown reasoning format)
            /* add_generation_prompt */ true,
            /* now_str           */ "",
            /* chat_template_kwargs */ {},
            /* force_pure_content*/ false
        );
        formatted = params.prompt;

        chat_parse_state state;
        state.parsable          = true;
        state.format            = static_cast<int>(params.format);
        state.reasoning_format  = COMMON_REASONING_FORMAT_AUTO;
        state.generation_prompt = params.generation_prompt;
        state.parser            = params.parser;
        // TEN-85: only meaningful when the template actually reasons — the
        // tags are left over from a previous variant otherwise, and a budget
        // built on them would arm against markers this prompt can't produce.
        if (params.supports_thinking) {
            state.thinking_start_tag = params.thinking_start_tag;
            state.thinking_end_tag   = params.thinking_end_tag;
        }
        // state.prompt is filled in below, from the round-tripped string.
        chat_state_put(rnctx, std::move(state));

        // Own try: common_chat_format_name() throws on a format it doesn't
        // know, and a llama.cpp bump adding one must not make a *log line* the
        // thing that silently drops every chat onto the legacy template path.
        const char* format_name = "unknown";
        try {
            format_name = common_chat_format_name(params.format);
        } catch (const std::exception&) { /* keep "unknown" */ }
        LOGI("nativeFormatChat: chat format=%s, parser=%zu bytes, generation_prompt=%zu chars, thinking=%d start='%s' end='%s'",
             format_name, params.parser.size(), params.generation_prompt.size(),
             params.supports_thinking, params.thinking_start_tag.c_str(), params.thinking_end_tag.c_str());
    } catch (const std::exception& e) {
        // Some templates fail under jinja — fall back to the legacy path
        // (no thinking control there).
        LOGE("nativeFormatChat: jinja failed (%s), falling back to legacy", e.what());
        // No common_chat_params on that path, so nothing to parse the reply
        // with: drop any state a previous call left behind rather than apply a
        // stale parser to this prompt's output.
        chat_state_erase(rnctx);
        try {
            formatted = rnctx->getFormattedChat(msgs, "");
        } catch (const std::exception& e2) {
            LOGE("nativeFormatChat: exception: %s", e2.what());
            return env->NewStringUTF("");
        }
    }

    LOGI("nativeFormatChat: formatted %zu chars (thinking=%d)", formatted.size(), enableThinking);

    jstring jformatted = env->NewStringUTF(formatted.c_str());
    // Record the prompt as nativeCompletion will actually see it, not as it was
    // built here: JNI's modified UTF-8 is not the identity round-trip on
    // characters outside the BMP (an emoji in the user's message comes back as
    // a 6-byte CESU-8 surrogate pair, not the 4-byte UTF-8 it went in as), and
    // the equality check that gates chat parsing would then fail — silently
    // disabling the reasoning split for exactly the conversations that contain
    // one.
    chat_state_set_prompt(rnctx, jstring_to_std(env, jformatted));
    return jformatted;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeFormatChat(
        JNIEnv* env,
        jobject /* thiz */,
        jlong   h,
        jstring messagesJson,
        jboolean enableThinking)
{
    // The two template paths above already catch std::exception, so this only
    // covers what they can't — a throw from chat_state_put/NewStringUTF after
    // them, or something that isn't a std::exception. "" is the empty prompt
    // the h == 0 and legacy-fallback-failed paths return.
    return jni_guard("nativeFormatChat",
                     [&] { return formatChat_impl(env, h, messagesJson, enableThinking); },
                     [&] { return env->NewStringUTF(""); });
}

// ---------------------------------------------------------------------------
// nativeTokenize
// ---------------------------------------------------------------------------

static jintArray tokenize_impl(
        JNIEnv* env,
        jlong   h,
        jstring text)
{
    if (h == 0L) return env->NewIntArray(0);
    auto* rnctx = to_ctx(h);

    std::string txt = jstring_to_std(env, text);
    // No media — plain text tokenisation
    rnllama::llama_rn_tokenize_result result = rnctx->tokenize(txt, {});

    const auto& tokens = result.tokens;
    jintArray arr = env->NewIntArray(static_cast<jsize>(tokens.size()));
    if (arr == nullptr) return env->NewIntArray(0);

    // llama_token is int32_t — safe to reinterpret as jint
    env->SetIntArrayRegion(arr, 0, static_cast<jsize>(tokens.size()),
                           reinterpret_cast<const jint*>(tokens.data()));
    return arr;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeTokenize(
        JNIEnv* env,
        jobject /* thiz */,
        jlong   h,
        jstring text)
{
    // Kotlin declares this IntArray, not IntArray? — an empty array, the same
    // answer the h == 0 path gives, rather than a null the caller can't hold.
    return jni_guard("nativeTokenize",
                     [&] { return tokenize_impl(env, h, text); },
                     [&] { return env->NewIntArray(0); });
}

// ---------------------------------------------------------------------------
// nativeKvCacheUsedCells
// ---------------------------------------------------------------------------
// llama.cpp b9769 does not expose llama_kv_self_used_cells() in the public
// C API. Strategy: read completion->n_past which tracks how many tokens have
// been evaluated into the KV cache for sequence 0. This is accurate for the
// single-sequence use case targeted by this bridge.
// Fallback: llama_memory_seq_pos_max(mem, 0) + 1 via public memory API.

extern "C" JNIEXPORT jint JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeKvCacheUsedCells(
        JNIEnv* /* env */,
        jobject /* thiz */,
        jlong h)
{
    if (h == 0L) return 0;
    auto* rnctx = to_ctx(h);

    if (rnctx->completion != nullptr) {
        return static_cast<jint>(rnctx->completion->n_past);
    }

    // Fallback via public memory API
    if (rnctx->ctx != nullptr) {
        auto* mem = llama_get_memory(rnctx->ctx);
        if (mem != nullptr) {
            llama_pos pos = llama_memory_seq_pos_max(mem, 0);
            if (pos >= 0) return static_cast<jint>(pos + 1);
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// nativeInterrupt
// ---------------------------------------------------------------------------

extern "C" JNIEXPORT void JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeInterrupt(
        JNIEnv* /* env */,
        jobject /* thiz */,
        jlong h)
{
    if (h == 0L) return;
    auto* rnctx = to_ctx(h);
    if (rnctx->completion != nullptr) {
        rnctx->completion->is_interrupted = true;
        LOGI("nativeInterrupt: is_interrupted set");
    }
}

// ---------------------------------------------------------------------------
// nativeSaveSession
// ---------------------------------------------------------------------------
// Wraps llama_state_save_file (b9769 public API).
// rn-llama has no dedicated saveSession() — we call llama.cpp directly on
// rnctx->ctx. Token list = completion->embd, clamped to completion->n_past so
// the file never claims more tokens than the KV cache it ships with actually
// holds (TEN-92, see the block in the body).
// Returns token count saved, or -1 on failure.

static jint saveSession_impl(
        JNIEnv* env,
        jlong   h,
        jstring path)
{
    if (h == 0L) return -1;
    auto* rnctx = to_ctx(h);
    if (rnctx->ctx == nullptr) return -1;

    std::string spath = jstring_to_std(env, path);

    const std::vector<llama_token>* embd =
        (rnctx->completion != nullptr) ? &rnctx->completion->embd : nullptr;

    // -----------------------------------------------------------------
    // Clamp the token list to what the KV cache actually holds (TEN-92)
    // -----------------------------------------------------------------
    // embd is the token *list*; n_past is how many of them were decoded into
    // the cache. llama_state_save_file() writes both the cache and the count
    // it is handed, and never confronts the two — so whatever is passed here
    // is what the file claims, true or not.
    //
    // They diverge, in rn-completion.cpp's nextToken():
    //   ended by EOS         — no gap, it returns before the embd.push_back()
    //   ended by a stop word
    //   or by n_predict      — one token: the last one is sampled and pushed,
    //                          and generation stops before it is ever decoded
    //   aborted mid-graph    — everything left to decode, up to a whole prompt
    //     (decode_ret == 2)    on a cancelled prefill: that branch advances
    //                          neither n_past nor embd, unlike the
    //                          is_interrupted branch below it which resizes
    //                          embd back down to n_past
    //
    // Saving the larger number is what corrupts the restore. nativeLoadSession
    // sets n_past = N from the file while the cache holds N-1 cells; the next
    // prefill's find_common_prefix_length() answers N, so
    // llama_memory_seq_rm(kv, 0, N, -1) removes nothing, and
    // llama_batch_get_one() carries no explicit positions — they are derived
    // from seq_pos_max + 1, which is N-1. embd[N-1] drops out of the model's
    // view and everything after it shifts by one. No error, no log, and it
    // compounds with every save.
    //
    // min(), not n_past outright, and not a fix in rn-completion.cpp: that
    // file is vendored, and n_past is the honest floor in all three cases
    // anyway. The decode_ret == 2 comment there says n_past "matches or
    // undershoots what's actually committed", so clamping can leave real cells
    // undeclared — which is the harmless direction, because the next prefill's
    // seq_rm(0, n_past, -1) drops exactly those. Over-declaring is the one that
    // cannot be recovered from.
    size_t n_tokens = embd ? embd->size() : 0;
    if (embd != nullptr) {
        const size_t n_cached = static_cast<size_t>(rnctx->completion->n_past);
        if (n_cached < n_tokens) {
            LOGI("nativeSaveSession: %zu tokens sampled but %zu decoded — saving %zu",
                 n_tokens, n_cached, n_cached);
            n_tokens = n_cached;
        } else if (n_cached > n_tokens) {
            // Never observed and no path in rn-completion.cpp produces it:
            // loadPrompt() derives n_past from a prefix of embd, the context
            // shift drops the same count from both, and nativeLoadSession sets
            // them equal. If this fires, the cache is ahead of the token list
            // and the file cannot describe it — a defect of its own, not this
            // one. Saving embd->size() is the conservative read either way.
            LOGE("nativeSaveSession: n_past=%zu exceeds embd=%zu — token list is "
                 "behind the KV cache, saving %zu", n_cached, n_tokens, n_tokens);
        }
    }

    const llama_token* tokens = (n_tokens > 0) ? embd->data() : nullptr;

    bool ok = llama_state_save_file(rnctx->ctx, spath.c_str(), tokens, n_tokens);
    if (!ok) {
        LOGE("nativeSaveSession: llama_state_save_file failed, path=%s", spath.c_str());
        return -1;
    }
    LOGI("nativeSaveSession: saved %zu tokens to %s", n_tokens, spath.c_str());
    return static_cast<jint>(n_tokens);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeSaveSession(
        JNIEnv* env,
        jobject /* thiz */,
        jlong   h,
        jstring path)
{
    // -1 is this call's documented failure value; a session that couldn't be
    // saved is refused, not silently reported as empty.
    return jni_guard("nativeSaveSession",
                     [&] { return saveSession_impl(env, h, path); },
                     [] { return (jint) -1; });
}

// ---------------------------------------------------------------------------
// nativeLoadSession
// ---------------------------------------------------------------------------
// Wraps llama_state_load_file (b9769 public API).
// Returns token count loaded, or -1 on failure.

// Forces the one state that is always consistent after a restore that did not
// finish: empty KV cache, empty embd, so the next prefill is a full one. See
// the long comment at its call site below for why a half-restored cache is
// worse than no cache at all.
static void reset_after_failed_session_load(rnllama::llama_rn_context* rnctx) {
    if (rnctx->completion != nullptr) {
        rnctx->completion->embd.clear();
        rnctx->completion->n_past = 0;
    }
    auto* mem = llama_get_memory(rnctx->ctx);
    if (mem != nullptr) {
        llama_memory_clear(mem, false);
    }
}

static jint loadSession_impl(
        JNIEnv* env,
        jlong   h,
        jstring path)
{
    if (h == 0L) return -1;
    auto* rnctx = to_ctx(h);
    if (rnctx->ctx == nullptr) return -1;

    std::string spath = jstring_to_std(env, path);

    // Allocate buffer sized to full context window
    uint32_t n_ctx = static_cast<uint32_t>(rnctx->n_ctx > 0 ? rnctx->n_ctx : 4096);
    std::vector<llama_token> tokens_out(n_ctx);
    size_t n_token_count = 0;

    bool ok = llama_state_load_file(
        rnctx->ctx,
        spath.c_str(),
        tokens_out.data(),
        tokens_out.size(),
        &n_token_count
    );

    if (!ok) {
        // llama_state_load_file can fail *after* wiping the KV cache:
        // llama_kv_cache::state_read_meta() calls clear(true) before reading
        // cells back, and both failure exits below it leave the cache empty or
        // half-populated. Returning -1 as-is would keep embd/n_past describing
        // the previous conversation, and the next prefill would trust
        // find_common_prefix_length() over cells that no longer exist:
        // silently wrong attention, no crash, no log. So force the one state
        // that is always consistent: empty cache, empty embd. Next prefill is
        // then a full one, the fallback SessionRepository.restore expects.
        //
        // clear over seq_rm: the whole-cache restore path writes cells for
        // every seq_id in the file, not just 0, so seq_rm(0, -1, -1) could
        // leave foreign cells behind, and it can report failure on
        // recurrent/hybrid models. data=false clears metadata only, the data
        // buffers are unreachable once no cell refers to them (same choice
        // rn-completion makes on its own cache-clear fallback).
        //
        // This also clears on failures that happened before the cache was
        // touched (bad magic, truncated header): the public API returns a plain
        // bool, so the two are indistinguishable without patching vendored
        // llama-context.cpp. Cost is one extra prefill on a path where the
        // session file is discarded anyway.
        reset_after_failed_session_load(rnctx);

        LOGE("nativeLoadSession: llama_state_load_file failed, path=%s "
             "(KV cache and completion state reset, next prefill will be full)",
             spath.c_str());
        return -1;
    }

    // Sync completion context so nativeKvCacheUsedCells stays accurate
    if (rnctx->completion != nullptr) {
        tokens_out.resize(n_token_count);
        rnctx->completion->embd  = std::move(tokens_out);
        rnctx->completion->n_past = static_cast<llama_pos>(n_token_count);
    }

    LOGI("nativeLoadSession: loaded %zu tokens from %s", n_token_count, spath.c_str());
    return static_cast<jint>(n_token_count);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeLoadSession(
        JNIEnv* env,
        jobject /* thiz */,
        jlong   h,
        jstring path)
{
    // A throw out of llama_state_load_file() leaves exactly the half-restored
    // cache the `!ok` branch exists to clean up — same reset here, or the next
    // prefill would trust cells that describe a conversation nobody is having.
    return jni_guard("nativeLoadSession",
                     [&] { return loadSession_impl(env, h, path); },
                     [&] {
                         if (h != 0L) {
                             auto* rnctx = to_ctx(h);
                             if (rnctx->ctx != nullptr) reset_after_failed_session_load(rnctx);
                         }
                         return (jint) -1;
                     });
}
