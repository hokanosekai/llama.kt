# llama.kt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Une lib Android autonome (`llama.kt`) qui charge un GGUF et streame des tokens on-device via JNI sur llama.cpp, avec GPU OpenCL, extraite de la glue C++ de llama.rn, exposant une API Kotlin propre + les shims V1 (context budget, abort).

**Architecture:** Module Android library (produit un AAR). Le C++ de llama.rn est vendorisé via `bootstrap.sh` (copie llama.cpp du submodule + rewrite symboles `LM_`). On garde `cpp/rn-llama.*`, `rn-completion.*`, `ggml-opencl/`; on jette `cpp/jsi/` (adapter React Native). On écrit un `tensai_jni.cpp` (JNI standard, zéro dépendance RN) + un `LlamaEngine.kt` (bindings + loader `.so` + wrapper coroutines/Flow). Consommé par l'app tensai en git submodule.

**Tech Stack:** Kotlin, Android NDK (r27+), CMake, JNI, llama.cpp (submodule), OpenCL (Adreno), Gradle (android-library). Licence Apache 2.0.

**Vérification:** par build/run (pas de TDD — code natif device-dépendant). Un module `example/` (app démo minimale) sert de banc de validation manuel : charger un GGUF, générer, observer tok/s + backend.

**Specs source (vault):** `projects/tensai/spec-v1-runtime-chat.md`, `research-llama-rn-extraction.md`. PoC d'extraction existant : `ForbiddenByte/llama4aj` (`ajllamaJNI.cpp`).

---

### Task 1 : Repo + structure + remote

**Files:**
- Create: `settings.gradle.kts`, `build.gradle.kts`, `gradle.properties`, `.gitignore`, `LICENSE`, `THIRD_PARTY_LICENSES.md`, `README.md`
- Create: `llama-kt/build.gradle.kts` (module lib)

- [ ] **Step 1 : Init repo + remote**

```bash
mkdir -p ~/ghq/github.com/hokanosekai/llama.kt && cd $_
git init
gh repo create hokanosekai/llama.kt --public --source=. --remote=origin --description "Kotlin/Android JNI binding for llama.cpp — GGUF on-device inference, extracted from llama.rn"
```

- [ ] **Step 2 : Structure Gradle (android library)**

`settings.gradle.kts` :
```kotlin
pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement { repositories { google(); mavenCentral() } }
rootProject.name = "llama.kt"
include(":llama-kt", ":example")
```

`llama-kt/build.gradle.kts` (essentiel) :
```kotlin
plugins { id("com.android.library"); id("org.jetbrains.kotlin.android") }
android {
    namespace = "com.tensai.llamakt"; compileSdk = 35
    defaultConfig {
        minSdk = 29
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake { arguments += listOf("-DANDROID_STL=c++_shared") } }
    }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt") } }
    ndkVersion = "27.2.12479018"
}
kotlin { jvmToolchain(17) }
dependencies { implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0") }
```

- [ ] **Step 3 : Licences**

`LICENSE` = Apache 2.0 (texte complet). `THIRD_PARTY_LICENSES.md` liste : llama.rn (MIT, Jhen-Jie Hong), llama.cpp (MIT, ggml-org), OpenCL-ICD-Loader (Apache 2.0, Khronos).

- [ ] **Step 4 : Commit**

```bash
git add -A && git commit -m "chore: scaffold llama.kt android library module" && git push -u origin main
```

---

### Task 2 : Submodules + vendoring du source

**Files:**
- Create: `.gitmodules`, `scripts/bootstrap.sh`, `scripts/build-opencl.sh`
- Create (généré) : `llama-kt/src/main/cpp/` (contenu vendorisé)

- [ ] **Step 1 : Ajouter les submodules**

```bash
git submodule add https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git submodule add https://github.com/KhronosGroup/OpenCL-ICD-Loader third_party/OpenCL-ICD-Loader
git submodule add https://github.com/KhronosGroup/OpenCL-Headers third_party/OpenCL-Headers
cd third_party/llama.cpp && git checkout b9769 && cd ../..   # commit pinné (aligné llama.rn 0.12.5) — ajuster au besoin
```

- [ ] **Step 2 : Adapter bootstrap.sh depuis llama.rn**

Cloner llama.rn en référence, récupérer `scripts/bootstrap.sh`, RETIRER les étapes spécifiques React Native (copie vers les dossiers RN). Garder : copie `third_party/llama.cpp/{src,include,ggml,common,tools/mtmd}` vers `llama-kt/src/main/cpp/`, + rewrite sed des symboles (ggml_ vers lm_ggml_, GGML_ vers LM_GGML_).

```bash
git clone --depth 1 https://github.com/mybigday/llama.rn /tmp/llama.rn-ref
cp /tmp/llama.rn-ref/scripts/bootstrap.sh scripts/bootstrap.sh
# éditer : retirer les cp vers android/RN, garder la copie cpp/ + les sed de prefix
```

- [ ] **Step 3 : Récupérer la glue C++ RN-agnostique**

Copier depuis `/tmp/llama.rn-ref/cpp/` vers `llama-kt/src/main/cpp/` : `rn-llama.{h,cpp}`, `rn-completion.{h,cpp}`, `rn-slot*.{h,cpp}`, `rn-common.hpp`, `rn-mtmd.hpp`, `jsi/JSINativeHeaders.h`, `jsi/ThreadPool.{h,cpp}`. NE PAS copier le reste de `cpp/jsi/` (adapter RN). Vérifier via grep qu'aucun fichier gardé n'inclut les headers JSI/fbjni :

```bash
grep -rl "jsi/jsi.h\|fbjni" llama-kt/src/main/cpp/ && echo "RESTE DU RN — a virer" || echo "clean"
```

- [ ] **Step 4 : build-opencl.sh (stub ICD)**

Copier `scripts/build-opencl.sh` de llama.rn verbatim (build `libOpenCL.so` depuis les submodules Khronos vers `llama-kt/src/main/jniLibs/arm64-v8a/`).

- [ ] **Step 5 : Lancer bootstrap + commit**

```bash
bash scripts/bootstrap.sh && bash scripts/build-opencl.sh
git add -A && git commit -m "build: vendor llama.cpp + rn-llama glue via bootstrap, opencl stub"
```

Vérif : `llama-kt/src/main/cpp/` contient les sources préfixées LM_, pas de résidu JSI.

---

### Task 3 : CMakeLists (sans ReactAndroid)

**Files:**
- Create: `llama-kt/src/main/cpp/CMakeLists.txt`

- [ ] **Step 1 : Écrire le CMake**

S'inspirer de `llama.rn/android/src/main/rnllama/CMakeLists.txt` (build moteur, PAS le top-level RN-couplé). Structure :
```cmake
cmake_minimum_required(VERSION 3.22)
project(llamakt)
set(CMAKE_CXX_STANDARD 17)

file(GLOB LLAMA_SRC "*.cpp" "ggml-cpu/*.cpp" "ggml-opencl/*.cpp" "tools/mtmd/*.cpp" "common/*.cpp")
add_library(llamakt SHARED ${LLAMA_SRC} tensai_jni.cpp)

target_compile_definitions(llamakt PRIVATE
    LM_GGML_USE_CPU
    LM_GGML_USE_OPENCL
    LM_GGML_OPENCL_USE_ADRENO_KERNELS
    LM_GGML_OPENCL_EMBED_KERNELS
    LM_GGML_OPENCL_SOA_Q)

target_include_directories(llamakt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} common tools/mtmd)
find_library(LOG_LIB log)
target_link_libraries(llamakt ${LOG_LIB} android
    ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libOpenCL.so)
```
(Ajuster les globs exacts après avoir vu l'arbo réelle post-bootstrap.)

- [ ] **Step 2 : Vérifier que le C++ compile SANS le JNI**

Créer un `tensai_jni.cpp` temporaire minimal (`#include <jni.h>`), lancer un build gradle du module. Objectif : isoler les erreurs de compil du moteur avant d'écrire le JNI.

```bash
./gradlew :llama-kt:externalNativeBuildDebug
```
Attendu : le moteur compile (erreurs de link JNI OK à ce stade).

- [ ] **Step 3 : Commit**

```bash
git add -A && git commit -m "build: cmake for engine + opencl, no react-native deps"
```

---

### Task 4 : tensai_jni.cpp — pont JNI

**Files:**
- Create/Modify: `llama-kt/src/main/cpp/tensai_jni.cpp`

Référence : `ForbiddenByte/llama4aj/ajllamaJNI.cpp` (entry points appelant `rnllama::llama_rn_context`). On expose plus large.

- [ ] **Step 1 : Entry points cœur (load / free)**

```cpp
#include <jni.h>
#include <string>
#include "rn-llama.h"
using namespace rnllama;

extern "C" JNIEXPORT jlong JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeLoadModel(JNIEnv* env, jobject, jstring path, jint nGpuLayers) {
    auto* ctx = new llama_rn_context();
    common_params params;
    const char* p = env->GetStringUTFChars(path, nullptr);
    params.model.path = p;
    params.n_gpu_layers = nGpuLayers;
    env->ReleaseStringUTFChars(path, p);
    if (!ctx->loadModel(params)) { delete ctx; return 0; }
    return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT void JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeFree(JNIEnv*, jobject, jlong h) {
    delete reinterpret_cast<llama_rn_context*>(h);
}
```
(Signatures `common_params`/`loadModel` à ajuster selon l'API réelle de `rn-llama.h` post-bootstrap.)

- [ ] **Step 2 : Streaming completion via callback Kotlin**

```cpp
extern "C" JNIEXPORT void JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeCompletion(
    JNIEnv* env, jobject thiz, jlong h, jstring prompt, jobject callback) {
    auto* ctx = reinterpret_cast<llama_rn_context*>(h);
    jclass cbClass = env->GetObjectClass(callback);
    jmethodID onToken = env->GetMethodID(cbClass, "onToken", "(Ljava/lang/String;)V");
    const char* p = env->GetStringUTFChars(prompt, nullptr);
    ctx->params.prompt = p;
    ctx->beginCompletion(); ctx->loadPrompt();
    while (ctx->has_next_token && !ctx->is_interrupted) {
        auto tok = ctx->doCompletion();
        jstring js = env->NewStringUTF(tok.text_to_send.c_str());
        env->CallVoidMethod(callback, onToken, js);
        env->DeleteLocalRef(js);
    }
    ctx->endCompletion();
    env->ReleaseStringUTFChars(prompt, p);
}
```
(Noms `beginCompletion`/`doCompletion`/`has_next_token` = ceux de `rn-completion` — vérifier verbatim sur le clone.)

- [ ] **Step 3 : Shims V1 — budget context + abort + tokenize**

```cpp
extern "C" JNIEXPORT jint JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeKvCacheUsedCells(JNIEnv*, jobject, jlong h) {
    auto* ctx = reinterpret_cast<llama_rn_context*>(h);
    return llama_kv_self_used_cells(ctx->ctx);   // nom exact API llama.cpp à confirmer selon version
}
extern "C" JNIEXPORT void JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeInterrupt(JNIEnv*, jobject, jlong h) {
    reinterpret_cast<llama_rn_context*>(h)->is_interrupted = true;   // abort coopératif
}
extern "C" JNIEXPORT jintArray JNICALL
Java_com_tensai_llamakt_LlamaEngine_nativeTokenize(JNIEnv* env, jobject, jlong h, jstring text) {
    auto* ctx = reinterpret_cast<llama_rn_context*>(h);
    const char* t = env->GetStringUTFChars(text, nullptr);
    auto toks = ctx->tokenize(t, false);
    env->ReleaseStringUTFChars(text, t);
    jintArray out = env->NewIntArray(toks.size());
    env->SetIntArrayRegion(out, 0, toks.size(), reinterpret_cast<jint*>(toks.data()));
    return out;
}
```
Note : `rn-completion` gère déjà l'interruption via un flag — préférer le flag existant à `llama_set_abort_callback` s'il couvre le prefill ; sinon câbler `llama_set_abort_callback` dans le load.

- [ ] **Step 4 : Session save/load (persistance)**

Exposer `nativeSaveSession(h, path)` / `nativeLoadSession(h, path)` wrappant `ctx->saveSession(path)` / `loadSession(path)` (déjà dans rn-llama). Retour = nb tokens.

- [ ] **Step 5 : Build + commit**

```bash
./gradlew :llama-kt:assembleDebug
git add -A && git commit -m "feat: jni bridge — load, streaming completion, tokenize, kv budget, interrupt, session"
```
Attendu : l'AAR se build sans erreur de link.

---

### Task 5 : LlamaEngine.kt — API Kotlin + loader + Flow

**Files:**
- Create: `llama-kt/src/main/java/com/tensai/llamakt/LlamaEngine.kt`
- Create: `llama-kt/src/main/java/com/tensai/llamakt/TokenCallback.kt`

- [ ] **Step 1 : Callback + loader .so**

```kotlin
package com.tensai.llamakt
fun interface TokenCallback { fun onToken(token: String) }
```

```kotlin
package com.tensai.llamakt
object NativeLib {
    @Volatile private var loaded = false
    fun ensure() { if (!loaded) { System.loadLibrary("llamakt"); loaded = true } }
}
```

- [ ] **Step 2 : Bindings external fun**

```kotlin
package com.tensai.llamakt
class LlamaEngine {
    private var handle: Long = 0
    fun load(path: String, nGpuLayers: Int = 0) {
        NativeLib.ensure(); handle = nativeLoadModel(path, nGpuLayers)
        require(handle != 0L) { "load failed: $path" }
    }
    fun tokenizeCount(text: String): Int = nativeTokenize(handle, text).size
    fun kvUsed(): Int = nativeKvCacheUsedCells(handle)
    fun interrupt() = nativeInterrupt(handle)
    fun saveSession(path: String): Int = nativeSaveSession(handle, path)
    fun loadSession(path: String): Int = nativeLoadSession(handle, path)
    fun free() { if (handle != 0L) { nativeFree(handle); handle = 0 } }

    private external fun nativeLoadModel(path: String, nGpuLayers: Int): Long
    private external fun nativeCompletion(h: Long, prompt: String, cb: TokenCallback)
    private external fun nativeTokenize(h: Long, text: String): IntArray
    private external fun nativeKvCacheUsedCells(h: Long): Int
    private external fun nativeInterrupt(h: Long)
    private external fun nativeSaveSession(h: Long, path: String): Int
    private external fun nativeLoadSession(h: Long, path: String): Int
    private external fun nativeFree(h: Long)
    internal fun completionRaw(prompt: String, cb: TokenCallback) = nativeCompletion(handle, prompt, cb)
}
```

- [ ] **Step 3 : Wrapper coroutines/Flow**

```kotlin
package com.tensai.llamakt
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.Dispatchers

fun LlamaEngine.decode(prompt: String): Flow<String> = callbackFlow {
    val cb = TokenCallback { trySend(it) }
    completionRaw(prompt, cb)   // bloquant : tourne sur le dispatcher aval
    close()
    awaitClose { interrupt() }
}.flowOn(Dispatchers.Default)
```

- [ ] **Step 4 : Build + commit**

```bash
./gradlew :llama-kt:assembleDebug
git add -A && git commit -m "feat: kotlin api — LlamaEngine + coroutine Flow decode"
```

---

### Task 6 : Module example/ — banc de validation (run réel)

**Files:**
- Create: `example/build.gradle.kts`, `example/src/main/java/.../MainActivity.kt`, manifest

- [ ] **Step 1 : App démo minimale**

App 1 écran : bouton "pick GGUF" (SAF), champ prompt, sortie texte. Au run : `engine.load(fd_path, nGpuLayers = 99)` puis `decode(prompt).collect { append(it) }`. Afficher tok/s (tokens / temps) + `kvUsed()`.

Manifest : `<uses-native-library android:name="libOpenCL.so" android:required="false"/>`.

- [ ] **Step 2 : Run sur device réel**

```bash
./gradlew :example:installDebug
```
Sur le tel : pousser un petit GGUF (ex Llama-3.2-1B q4_K_M), le sélectionner, prompt "Hello", observer :
- tokens qui streament (streaming OK)
- tok/s affiché (perf sanity)
- backend GPU actif si Adreno (logcat `ggml-opencl`)
- `kvUsed()` qui monte (shim budget OK)
- bouton stop qui coupe (interrupt OK)

- [ ] **Step 3 : Commit**

```bash
git add -A && git commit -m "feat: example app — manual validation bench (load, stream, tok/s, gpu, interrupt)"
git push
```

**Critère de succès du plan :** l'app example charge un GGUF arbitraire et streame des tokens on-device, GPU actif sur Adreno, stop fonctionnel. Alors llama.kt est prouvé, on passe au plan de l'app tensai (qui le consomme en submodule).

---

## Notes d'exécution

- **Plan B (contingence, pas une tâche) :** si Task 2-4 bloquent dur (glue llama.rn inextricable), fallback = repartir de l'exemple officiel `llama.cpp/examples/llama.android` (JNI thin, MIT) et y greffer OpenCL + shims. Plus de boulot GPU, mais débloque.
- **Gaps à confirmer sur le clone (étapes réelles d'adaptation, pas des placeholders) :** noms exacts API `rn-llama.h`/`rn-completion` (`loadModel`/`doCompletion`/`is_interrupted`/`tokenize`/`saveSession`) ; nom exact du call llama.cpp used-cells (`llama_kv_self_used_cells` vs `llama_kv_cache_used_cells` selon version pinnée) ; globs exacts du CMake post-bootstrap.
- **Ce plan vit hors vault** (code-heavy) ; à déplacer dans `llama.kt/docs/` une fois le repo créé (Task 1).
