/**
 * tensai_jni.cpp — JNI bridge for com.tensai.llamakt.LlamaEngine
 *
 * Entry points:
 *   nativeLoadModel, nativeFree, nativeCompletion,
 *   nativeFormatChat, nativeTokenize, nativeKvCacheUsedCells,
 *   nativeInterrupt, nativeSaveSession, nativeLoadSession,
 *   nativeListBackends, nativeActiveBackend
 *
 * KV used-cells strategy: this version of llama.cpp (b9769) exposes no
 * public llama_kv_self_used_cells() or llama_kv_self_n_tokens(). The
 * closest public measure is llama_memory_seq_pos_max(mem, 0) + 1, which
 * equals n_past for sequence 0. We prefer completion->n_past directly
 * when available (always alive after loadModel succeeds).
 */

#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>

#include "rn-llama.h"
#include "rn-completion.h"
#include "rn-common.hpp"
#include "common/common.h"
#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <sys/stat.h>

#define LOG_TAG "TensaiJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

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

// ---------------------------------------------------------------------------
// nativeListBackends  (static — no model required)
// ---------------------------------------------------------------------------
// Returns a JSON array describing all available ggml backend devices.
// Uses the existing backend_devices_info() from rn-common.hpp which
// enumerates lm_ggml_backend_dev_count() devices and collects:
//   backend  — backend registry name (e.g. "Vulkan", "CPU", "OpenCL")
//   type     — "cpu" | "gpu" | "igpu" | "accel"
//   deviceName — human-readable device name (e.g. "Mali-G68 MC4")

extern "C" JNIEXPORT jstring JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeListBackends(
        JNIEnv* env,
        jobject /* thiz */)
{
    std::string info = rnllama::backend_devices_info();
    LOGI("nativeListBackends: %s", info.c_str());
    return env->NewStringUTF(info.c_str());
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

extern "C" JNIEXPORT jstring JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeActiveBackend(
        JNIEnv* env,
        jobject /* thiz */,
        jlong h)
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

// ---------------------------------------------------------------------------
// nativeReadGgufMetadata  (static — reads the GGUF header only, no weights)
// ---------------------------------------------------------------------------
// Returns a JSON string with architecture, name, file_type, context_length,
// embedding_length, block_count, param_count, vocab_size, file_size_bytes.
// no_alloc=true → only the key-value header and tensor infos are read; a
// multi-GB model is inspected in milliseconds. Returns nullptr on failure.

extern "C" JNIEXPORT jstring JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeReadGgufMetadata(
        JNIEnv* env,
        jclass /* clazz */,
        jstring path)
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

    struct stat st{};
    const uint64_t file_size = (stat(fpath.c_str(), &st) == 0)
        ? static_cast<uint64_t>(st.st_size) : 0;

    nlohmann::json j = {
        {"architecture",     arch},
        {"name",             get_str("general.name")},
        {"file_type",        get_uint("general.file_type")},
        {"context_length",   get_uint(arch + ".context_length")},
        {"embedding_length", get_uint(arch + ".embedding_length")},
        {"block_count",      get_uint(arch + ".block_count")},
        {"param_count",      n_params},
        {"vocab_size",       vocab},
        {"file_size_bytes",  file_size},
        {"supports_thinking", supports_thinking},
    };

    lm_gguf_free(gctx);

    const std::string out = j.dump();
    LOGI("nativeReadGgufMetadata: %s", out.c_str());
    return env->NewStringUTF(out.c_str());
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
    auto* rnctx = new rnllama::llama_rn_context();

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
        delete rnctx;
        return 0L;
    }

    LOGI("loadModel OK, ptr=%p", rnctx);
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(rnctx));
}

// ---------------------------------------------------------------------------
// nativeFree
// ---------------------------------------------------------------------------

extern "C" JNIEXPORT void JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeFree(
        JNIEnv* /* env */,
        jobject /* thiz */,
        jlong h)
{
    if (h == 0L) return;
    auto* rnctx = to_ctx(h);
    LOGI("nativeFree ptr=%p", rnctx);
    delete rnctx;
}

// ---------------------------------------------------------------------------
// nativeCompletion
// ---------------------------------------------------------------------------
// Kotlin interface attendue : interface TokenCallback { fun onToken(token: String) }

extern "C" JNIEXPORT void JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeCompletion(
        JNIEnv* env,
        jobject /* thiz */,
        jlong   h,
        jstring prompt,
        jint    nPredict,
        jfloat  temperature,
        jint    topK,
        jfloat  topP,
        jfloat  minP,
        jobjectArray stopSequences,
        jobject cb)
{
    if (h == 0L) return;
    auto* rnctx = to_ctx(h);

    if (rnctx->completion == nullptr) {
        LOGE("nativeCompletion: completion context is null");
        return;
    }

    // Resolve callback method once
    jclass cbClass = env->GetObjectClass(cb);
    jmethodID onToken = env->GetMethodID(cbClass, "onToken", "(Ljava/lang/String;)V");
    if (onToken == nullptr) {
        LOGE("nativeCompletion: onToken method not found");
        env->DeleteLocalRef(cbClass);
        return;
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

    if (!comp->initSampling()) {
        LOGE("nativeCompletion: initSampling failed");
        env->DeleteLocalRef(cbClass);
        return;
    }

    // Tokenise and load prompt (no media)
    comp->loadPrompt({});

    // Begin generation
    comp->beginCompletion();

    auto emit = [&](const std::string& s) {
        if (s.empty()) return;
        jstring jtok = env->NewStringUTF(s.c_str());
        env->CallVoidMethod(cb, onToken, jtok);
        env->DeleteLocalRef(jtok);
    };

    // With stop sequences, tokens are staged in `pending` and only emitted
    // once they can no longer be part of a stop word (a stop can span
    // several tokens). On a full match, everything from the match start
    // is dropped so the stream never contains the stop sequence.
    std::string pending;

    while (comp->has_next_token && !comp->is_interrupted) {
        rnllama::completion_token_output tok_out = comp->doCompletion();

        if (tok_out.tok == -1) {
            break;
        }

        const std::string& text = tok_out.text;

        if (!has_stops) {
            emit(text);
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
    }

    // Generation ended without a stop match (EOS / n_predict / interrupt):
    // flush whatever was held back.
    if (!pending.empty() && !comp->is_interrupted) {
        emit(pending);
    }

    env->DeleteLocalRef(cbClass);
}

// ---------------------------------------------------------------------------
// nativeFormatChat
// ---------------------------------------------------------------------------
// Applies the model's built-in chat template (from GGUF metadata) to a JSON
// array of messages: [{"role":"user","content":"..."},...]
// Passing an empty chat_template string uses the template embedded in the GGUF.
// Returns the fully formatted prompt string ready for completion.

extern "C" JNIEXPORT jstring JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeFormatChat(
        JNIEnv* env,
        jobject /* thiz */,
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
        formatted = rnctx->getFormattedChatWithJinja(
            msgs,
            /* chat_template     */ "",
            /* json_schema       */ "",
            /* tools             */ "",
            /* parallel_tool_call*/ false,
            /* tool_choice       */ "",
            /* enable_thinking   */ enableThinking == JNI_TRUE,
            /* reasoning_format  */ "none",  // "" throws (Unknown reasoning format); none = raw passthrough
            /* add_generation_prompt */ true,
            /* now_str           */ "",
            /* chat_template_kwargs */ {},
            /* force_pure_content*/ false
        ).prompt;
    } catch (const std::exception& e) {
        // Some templates fail under jinja — fall back to the legacy path
        // (no thinking control there).
        LOGE("nativeFormatChat: jinja failed (%s), falling back to legacy", e.what());
        try {
            formatted = rnctx->getFormattedChat(msgs, "");
        } catch (const std::exception& e2) {
            LOGE("nativeFormatChat: exception: %s", e2.what());
            return env->NewStringUTF("");
        }
    }

    LOGI("nativeFormatChat: formatted %zu chars (thinking=%d)", formatted.size(), enableThinking);
    return env->NewStringUTF(formatted.c_str());
}

// ---------------------------------------------------------------------------
// nativeTokenize
// ---------------------------------------------------------------------------

extern "C" JNIEXPORT jintArray JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeTokenize(
        JNIEnv* env,
        jobject /* thiz */,
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
// rnctx->ctx. Token list = completion->embd (evaluated prompt tokens).
// Returns token count saved, or -1 on failure.

extern "C" JNIEXPORT jint JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeSaveSession(
        JNIEnv* env,
        jobject /* thiz */,
        jlong   h,
        jstring path)
{
    if (h == 0L) return -1;
    auto* rnctx = to_ctx(h);
    if (rnctx->ctx == nullptr) return -1;

    std::string spath = jstring_to_std(env, path);

    const std::vector<llama_token>* embd =
        (rnctx->completion != nullptr) ? &rnctx->completion->embd : nullptr;

    const llama_token* tokens = (embd && !embd->empty()) ? embd->data() : nullptr;
    size_t n_tokens = embd ? embd->size() : 0;

    bool ok = llama_state_save_file(rnctx->ctx, spath.c_str(), tokens, n_tokens);
    if (!ok) {
        LOGE("nativeSaveSession: llama_state_save_file failed, path=%s", spath.c_str());
        return -1;
    }
    LOGI("nativeSaveSession: saved %zu tokens to %s", n_tokens, spath.c_str());
    return static_cast<jint>(n_tokens);
}

// ---------------------------------------------------------------------------
// nativeLoadSession
// ---------------------------------------------------------------------------
// Wraps llama_state_load_file (b9769 public API).
// Returns token count loaded, or -1 on failure.

extern "C" JNIEXPORT jint JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeLoadSession(
        JNIEnv* env,
        jobject /* thiz */,
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
        LOGE("nativeLoadSession: llama_state_load_file failed, path=%s", spath.c_str());
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
