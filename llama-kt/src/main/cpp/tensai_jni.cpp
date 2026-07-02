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

#define LOG_TAG "TensaiJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

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
// nativeLoadModel
// ---------------------------------------------------------------------------

extern "C" JNIEXPORT jlong JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeLoadModel(
        JNIEnv* env,
        jobject /* thiz */,
        jstring path,
        jint nGpuLayers,
        jint nCtx)
{
    auto* rnctx = new rnllama::llama_rn_context();

    common_params p;
    p.model.path   = jstring_to_std(env, path);
    p.n_gpu_layers = static_cast<int32_t>(nGpuLayers);
    p.n_ctx        = static_cast<int32_t>(nCtx > 0 ? nCtx : 4096);
    if (p.n_batch <= 0) p.n_batch = 512;

    LOGI("loadModel: path=%s n_gpu_layers=%d n_ctx=%d n_batch=%d",
         p.model.path.c_str(), p.n_gpu_layers, p.n_ctx, p.n_batch);

    if (!rnctx->loadModel(p)) {
        LOGE("loadModel failed");
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
    // temp <= 0 → greedy decoding; topK 0 / topP 1.0 disable those filters.
    rnctx->params.sampling.temp  = static_cast<float>(temperature);
    rnctx->params.sampling.top_k = static_cast<int32_t>(topK);
    rnctx->params.sampling.top_p = static_cast<float>(topP);
    LOGI("nativeCompletion: n_predict=%d temp=%.2f top_k=%d top_p=%.2f",
         rnctx->params.n_predict, rnctx->params.sampling.temp,
         rnctx->params.sampling.top_k, rnctx->params.sampling.top_p);

    auto* comp = rnctx->completion;
    comp->rewind();

    if (!comp->initSampling()) {
        LOGE("nativeCompletion: initSampling failed");
        env->DeleteLocalRef(cbClass);
        return;
    }

    // Tokenise and load prompt (no media)
    comp->loadPrompt({});

    // Begin generation
    comp->beginCompletion();

    while (comp->has_next_token && !comp->is_interrupted) {
        rnllama::completion_token_output tok_out = comp->doCompletion();

        if (tok_out.tok == -1) {
            break;
        }

        const std::string& text = tok_out.text;
        if (!text.empty()) {
            jstring jtok = env->NewStringUTF(text.c_str());
            env->CallVoidMethod(cb, onToken, jtok);
            env->DeleteLocalRef(jtok);
        }
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
        jstring messagesJson)
{
    if (h == 0L) return env->NewStringUTF("");
    auto* rnctx = to_ctx(h);

    std::string msgs = jstring_to_std(env, messagesJson);

    std::string formatted;
    try {
        // Pass empty chat_template → use the template stored in the GGUF via
        // rnctx->templates (initialized from the model in loadModel).
        // add_generation_prompt defaults to true in common_chat_templates_inputs,
        // so the assistant prefix is automatically appended.
        formatted = rnctx->getFormattedChat(msgs, /* chat_template= */ "");
    } catch (const std::exception& e) {
        LOGE("nativeFormatChat: exception: %s", e.what());
        return env->NewStringUTF("");
    }

    LOGI("nativeFormatChat: formatted %zu chars", formatted.size());
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
