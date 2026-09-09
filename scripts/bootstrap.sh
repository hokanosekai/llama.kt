#!/bin/bash -e

# bootstrap.sh — vendorize llama.cpp sources + rn-llama glue into llama-kt/src/main/cpp/
# Adapted from mybigday/llama.rn scripts/bootstrap.sh:
#   - Target is llama-kt/src/main/cpp/ (not cpp/ at root, not android/ or ios/)
#   - No React Native / Hexagon SDK / iOS steps
#   - Adds rn-*.{h,cpp,hpp} glue + jsi/JSINativeHeaders.h + jsi/ThreadPool.{h,cpp}
#   - Skips ggml-metal (iOS only) and ggml-hexagon (not needed)
#   - Applies same LM_ prefix rewrite on ggml/gguf symbols

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OS=$(uname)

LLAMA_DIR="$ROOT_DIR/third_party/llama.cpp"
CPP_DIR="$ROOT_DIR/llama-kt/src/main/cpp"

# llama.rn is not a submodule (see scripts/llama.rn.rev for why): we clone it
# into a scratch dir and hard-check out the pinned revision. Override the
# location with LLAMA_RN_REF_DIR if /tmp is inconvenient.
LLAMA_RN_URL="https://github.com/mybigday/llama.rn"
LLAMA_RN_REV_FILE="$ROOT_DIR/scripts/llama.rn.rev"
LLAMA_RN_REF_DIR="${LLAMA_RN_REF_DIR:-/tmp/llama.rn-ref}"
LLAMA_RN_CPP="$LLAMA_RN_REF_DIR/cpp"
PATCHES_SRC="$LLAMA_RN_REF_DIR/scripts/patches"

echo "==> Init llama.cpp submodule..."
git -C "$ROOT_DIR" submodule init "third_party/llama.cpp"
# Use --no-fetch if the submodule is already at the right commit (faster).
# Otherwise fall back to a full update. We do NOT use --recursive to avoid
# pulling llama.cpp's own deps (ggml-cuda etc.) which we don't need.
LLAMA_WANT=$(git -C "$ROOT_DIR" ls-tree HEAD third_party/llama.cpp | awk '{print $3}')
LLAMA_HAVE=$(git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null || echo "none")
if [ "$LLAMA_WANT" != "$LLAMA_HAVE" ]; then
  git -C "$ROOT_DIR" submodule update "third_party/llama.cpp"
else
  echo "  submodule already at $LLAMA_HAVE, skipping update"
fi

# ---------------------------------------------------------------------------
# Pinned llama.rn checkout (rn-* glue + inherited patches)
# ---------------------------------------------------------------------------
# Done up front, not at the point of use: the rn-* copy happens ~400 lines
# below, after the Vulkan shader build, and nobody wants to find out the pin
# is unreachable twenty minutes in.
echo "==> Checking out pinned llama.rn..."
if [ ! -f "$LLAMA_RN_REV_FILE" ]; then
  echo "ERROR: missing pin file $LLAMA_RN_REV_FILE"
  exit 1
fi
LLAMA_RN_REV=$(grep -v '^#' "$LLAMA_RN_REV_FILE" | tr -d '[:space:]')
if ! echo "$LLAMA_RN_REV" | grep -qE '^[0-9a-f]{40}$'; then
  echo "ERROR: $LLAMA_RN_REV_FILE must hold one full 40-char llama.rn commit sha"
  echo "  got: '$LLAMA_RN_REV'"
  exit 1
fi

if [ ! -d "$LLAMA_RN_REF_DIR/.git" ]; then
  echo "  Cloning $LLAMA_RN_URL into $LLAMA_RN_REF_DIR..."
  rm -rf "$LLAMA_RN_REF_DIR"
  # Blobless, no checkout: we only need cpp/ and scripts/patches/ at one
  # revision. Deliberately not --recursive — llama.rn carries its own
  # llama.cpp submodule, which we already have in third_party/.
  git clone --filter=blob:none --no-checkout "$LLAMA_RN_URL" "$LLAMA_RN_REF_DIR"
fi

LLAMA_RN_HAVE=$(git -C "$LLAMA_RN_REF_DIR" rev-parse HEAD 2>/dev/null || echo "none")
if [ "$LLAMA_RN_HAVE" != "$LLAMA_RN_REV" ]; then
  echo "  Fetching $LLAMA_RN_REV..."
  git -C "$LLAMA_RN_REF_DIR" fetch --no-tags origin "$LLAMA_RN_REV" 2>/dev/null \
    || git -C "$LLAMA_RN_REF_DIR" fetch --no-tags --unshallow origin 2>/dev/null \
    || git -C "$LLAMA_RN_REF_DIR" fetch --no-tags origin
  git -C "$LLAMA_RN_REF_DIR" checkout --force --detach "$LLAMA_RN_REV"
fi

# The pin only means something if we verify it. A pre-existing checkout at
# $LLAMA_RN_REF_DIR (older bootstrap runs left a `--depth 1` clone of the
# default branch there) must not be used as-is.
LLAMA_RN_HAVE=$(git -C "$LLAMA_RN_REF_DIR" rev-parse HEAD)
if [ "$LLAMA_RN_HAVE" != "$LLAMA_RN_REV" ]; then
  echo "ERROR: $LLAMA_RN_REF_DIR is at $LLAMA_RN_HAVE, expected $LLAMA_RN_REV"
  exit 1
fi
if [ -n "$(git -C "$LLAMA_RN_REF_DIR" status --porcelain -- cpp scripts/patches)" ]; then
  echo "ERROR: $LLAMA_RN_REF_DIR has local edits under cpp/ or scripts/patches/."
  echo "  Vendored sources must come from a pristine $LLAMA_RN_REV checkout."
  exit 1
fi
echo "  llama.rn at $LLAMA_RN_REV"

echo "==> Creating output directory: $CPP_DIR"
mkdir -p "$CPP_DIR"
mkdir -p "$CPP_DIR/ggml-cpu/amx"
mkdir -p "$CPP_DIR/ggml-cpu/arch"
mkdir -p "$CPP_DIR/common/jinja"
mkdir -p "$CPP_DIR/tools/mtmd"
mkdir -p "$CPP_DIR/models"
mkdir -p "$CPP_DIR/ggml-opencl"
mkdir -p "$CPP_DIR/nlohmann"
mkdir -p "$CPP_DIR/jsi"
mkdir -p "$CPP_DIR/ggml-vulkan/shaders"
mkdir -p "$CPP_DIR/vulkan-hpp/vulkan"
mkdir -p "$CPP_DIR/spirv-headers/spirv/unified1"

# ---------------------------------------------------------------------------
# ggml public headers
# ---------------------------------------------------------------------------
echo "==> Copying ggml headers..."
cp "$LLAMA_DIR/ggml/include/ggml.h"            "$CPP_DIR/ggml.h"
cp "$LLAMA_DIR/ggml/include/ggml-alloc.h"      "$CPP_DIR/ggml-alloc.h"
cp "$LLAMA_DIR/ggml/include/ggml-backend.h"    "$CPP_DIR/ggml-backend.h"
cp "$LLAMA_DIR/ggml/include/ggml-cpu.h"        "$CPP_DIR/ggml-cpu.h"
cp "$LLAMA_DIR/ggml/include/ggml-cpp.h"        "$CPP_DIR/ggml-cpp.h"
cp "$LLAMA_DIR/ggml/include/ggml-opt.h"        "$CPP_DIR/ggml-opt.h"
cp "$LLAMA_DIR/ggml/include/ggml-opencl.h"     "$CPP_DIR/ggml-opencl.h"
cp "$LLAMA_DIR/ggml/include/gguf.h"            "$CPP_DIR/gguf.h"
# ggml-metal.h + ggml-blas.h + ggml-hexagon.h skipped (iOS / blas / hexagon only)

# ---------------------------------------------------------------------------
# ggml-opencl (Android GPU backend)
# ---------------------------------------------------------------------------
echo "==> Copying ggml-opencl..."
rm -rf "$CPP_DIR/ggml-opencl"
cp -r "$LLAMA_DIR/ggml/src/ggml-opencl" "$CPP_DIR/"
rm -f "$CPP_DIR/ggml-opencl/CMakeLists.txt"

# ---------------------------------------------------------------------------
# ggml-vulkan (Android Vulkan GPU backend — Mali, Adreno, etc.)
# ---------------------------------------------------------------------------
echo "==> Setting up ggml-vulkan..."

# 1. Copy ggml-vulkan.cpp and ggml-vulkan.h from submodule
cp "$LLAMA_DIR/ggml/src/ggml-vulkan/ggml-vulkan.cpp" "$CPP_DIR/ggml-vulkan/"
cp "$LLAMA_DIR/ggml/include/ggml-vulkan.h"            "$CPP_DIR/ggml-vulkan/"

# 2. Build vulkan-shaders-gen for host and generate SPIR-V shaders
#    Prerequisites: cmake, glslc (from shaderc/glslang), g++
VK_SHADERS_SRC="$LLAMA_DIR/ggml/src/ggml-vulkan/vulkan-shaders"
VK_BUILD_DIR="/tmp/llama-kt-vk-shaders-build"
VK_SPV_DIR="/tmp/llama-kt-vk-shaders-spv"
VK_OUT_DIR="/tmp/llama-kt-vk-shaders-out"

mkdir -p "$VK_BUILD_DIR" "$VK_SPV_DIR" "$VK_OUT_DIR"

echo "  Building vulkan-shaders-gen for host..."
cmake -S "$VK_SHADERS_SRC" -B "$VK_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -Wno-dev > /dev/null 2>&1
cmake --build "$VK_BUILD_DIR" --config Release -j$(nproc) > /dev/null 2>&1
VK_GEN="$VK_BUILD_DIR/vulkan-shaders-gen"

if [ ! -x "$VK_GEN" ]; then
  echo "ERROR: vulkan-shaders-gen build failed — cannot generate Vulkan shaders"
  exit 1
fi

echo "  Generating ggml-vulkan-shaders.hpp header..."
"$VK_GEN" \
  --output-dir "$VK_SPV_DIR" \
  --target-hpp "$VK_OUT_DIR/ggml-vulkan-shaders.hpp"

echo "  Compiling GLSL shaders to SPIR-V and generating per-shader .cpp files..."
for comp in "$VK_SHADERS_SRC"/*.comp; do
  base=$(basename "$comp")
  "$VK_GEN" \
    --glslc "$(which glslc)" \
    --source "$comp" \
    --output-dir "$VK_SPV_DIR" \
    --target-hpp "$VK_OUT_DIR/ggml-vulkan-shaders.hpp" \
    --target-cpp "$VK_OUT_DIR/${base}.cpp"
done

cp "$VK_OUT_DIR/ggml-vulkan-shaders.hpp" "$CPP_DIR/ggml-vulkan/"

# Fix relative include in shader cpp files (they're compiled from shaders/ subdir)
for f in "$VK_OUT_DIR"/*.cpp; do
  sed 's|#include "ggml-vulkan-shaders.hpp"|#include "../ggml-vulkan-shaders.hpp"|g' "$f" \
    > "$CPP_DIR/ggml-vulkan/shaders/$(basename "$f")"
done

# 3. Bundle Vulkan C++ binding headers (vulkan.hpp — not in NDK sysroot)
VULKAN_HEADERS_DIR="/tmp/Vulkan-Headers"
if [ ! -d "$VULKAN_HEADERS_DIR" ]; then
  echo "  Cloning Vulkan-Headers for vulkan.hpp..."
  git clone --depth=1 --branch v1.4.350 https://github.com/KhronosGroup/Vulkan-Headers "$VULKAN_HEADERS_DIR"
fi
cp "$VULKAN_HEADERS_DIR/include/vulkan/"*.hpp "$CPP_DIR/vulkan-hpp/vulkan/" 2>/dev/null || true

# 4. Bundle SPIR-V headers (spirv/unified1/spirv.hpp)
SPIRV_HEADERS_DIR="/tmp/SPIRV-Headers"
if [ ! -d "$SPIRV_HEADERS_DIR" ]; then
  echo "  Cloning SPIRV-Headers..."
  git clone --depth=1 https://github.com/KhronosGroup/SPIRV-Headers "$SPIRV_HEADERS_DIR"
fi
cp "$SPIRV_HEADERS_DIR/include/spirv/unified1/spirv.hpp" "$CPP_DIR/spirv-headers/spirv/unified1/"

echo "  Vulkan setup complete ($(ls "$CPP_DIR/ggml-vulkan/shaders/" | wc -l) shaders)"

# ---------------------------------------------------------------------------
# ggml-cpu
# ---------------------------------------------------------------------------
echo "==> Copying ggml-cpu..."
cp "$LLAMA_DIR/ggml/src/ggml-cpu/ggml-cpu.c"         "$CPP_DIR/ggml-cpu/ggml-cpu.c"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/ggml-cpu.cpp"       "$CPP_DIR/ggml-cpu/ggml-cpu.cpp"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/ggml-cpu-impl.h"    "$CPP_DIR/ggml-cpu/ggml-cpu-impl.h"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/quants.h"           "$CPP_DIR/ggml-cpu/quants.h"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/quants.c"           "$CPP_DIR/ggml-cpu/quants.c"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/arch-fallback.h"    "$CPP_DIR/ggml-cpu/arch-fallback.h"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/repack.cpp"         "$CPP_DIR/ggml-cpu/repack.cpp"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/repack.h"           "$CPP_DIR/ggml-cpu/repack.h"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/traits.h"           "$CPP_DIR/ggml-cpu/traits.h"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/traits.cpp"         "$CPP_DIR/ggml-cpu/traits.cpp"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/common.h"           "$CPP_DIR/ggml-cpu/common.h"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/unary-ops.h"        "$CPP_DIR/ggml-cpu/unary-ops.h"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/unary-ops.cpp"      "$CPP_DIR/ggml-cpu/unary-ops.cpp"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/binary-ops.h"       "$CPP_DIR/ggml-cpu/binary-ops.h"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/binary-ops.cpp"     "$CPP_DIR/ggml-cpu/binary-ops.cpp"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/vec.h"              "$CPP_DIR/ggml-cpu/vec.h"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/vec.cpp"            "$CPP_DIR/ggml-cpu/vec.cpp"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/simd-mappings.h"    "$CPP_DIR/ggml-cpu/simd-mappings.h"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/simd-gemm.h"        "$CPP_DIR/ggml-cpu/simd-gemm.h"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/ops.h"              "$CPP_DIR/ggml-cpu/ops.h"
cp "$LLAMA_DIR/ggml/src/ggml-cpu/ops.cpp"            "$CPP_DIR/ggml-cpu/ops.cpp"
cp -r "$LLAMA_DIR/ggml/src/ggml-cpu/amx"             "$CPP_DIR/ggml-cpu/"
cp -r "$LLAMA_DIR/ggml/src/ggml-cpu/arch/arm"        "$CPP_DIR/ggml-cpu/arch/"
cp -r "$LLAMA_DIR/ggml/src/ggml-cpu/arch/x86"        "$CPP_DIR/ggml-cpu/arch/"

# ---------------------------------------------------------------------------
# ggml core sources
# ---------------------------------------------------------------------------
echo "==> Copying ggml core sources..."
cp "$LLAMA_DIR/ggml/src/ggml.c"                "$CPP_DIR/ggml.c"
cp "$LLAMA_DIR/ggml/src/ggml-impl.h"           "$CPP_DIR/ggml-impl.h"
cp "$LLAMA_DIR/ggml/src/ggml-alloc.c"          "$CPP_DIR/ggml-alloc.c"
cp "$LLAMA_DIR/ggml/src/ggml-backend.cpp"      "$CPP_DIR/ggml-backend.cpp"
cp "$LLAMA_DIR/ggml/src/ggml-backend-impl.h"   "$CPP_DIR/ggml-backend-impl.h"
cp "$LLAMA_DIR/ggml/src/ggml-backend-reg.cpp"  "$CPP_DIR/ggml-backend-reg.cpp"
cp "$LLAMA_DIR/ggml/src/ggml-backend-meta.cpp" "$CPP_DIR/ggml-backend-meta.cpp"
cp "$LLAMA_DIR/ggml/src/ggml-backend-dl.h"     "$CPP_DIR/ggml-backend-dl.h"
cp "$LLAMA_DIR/ggml/src/ggml-backend-dl.cpp"   "$CPP_DIR/ggml-backend-dl.cpp"
cp "$LLAMA_DIR/ggml/src/ggml-common.h"         "$CPP_DIR/ggml-common.h"
cp "$LLAMA_DIR/ggml/src/ggml-opt.cpp"          "$CPP_DIR/ggml-opt.cpp"
cp "$LLAMA_DIR/ggml/src/ggml-quants.h"         "$CPP_DIR/ggml-quants.h"
cp "$LLAMA_DIR/ggml/src/ggml-quants.c"         "$CPP_DIR/ggml-quants.c"
cp "$LLAMA_DIR/ggml/src/ggml-threading.cpp"    "$CPP_DIR/ggml-threading.cpp"
cp "$LLAMA_DIR/ggml/src/ggml-threading.h"      "$CPP_DIR/ggml-threading.h"
cp "$LLAMA_DIR/ggml/src/gguf.cpp"              "$CPP_DIR/gguf.cpp"

# ---------------------------------------------------------------------------
# llama API
# ---------------------------------------------------------------------------
echo "==> Copying llama API..."
cp "$LLAMA_DIR/include/llama.h"                       "$CPP_DIR/llama.h"
cp "$LLAMA_DIR/include/llama-cpp.h"                   "$CPP_DIR/llama-cpp.h"
rm -rf "$CPP_DIR/models"
cp -r "$LLAMA_DIR/src/models"                         "$CPP_DIR/models"
cp "$LLAMA_DIR/src/llama.cpp"                         "$CPP_DIR/llama.cpp"
cp "$LLAMA_DIR/src/llama-chat.h"                      "$CPP_DIR/llama-chat.h"
cp "$LLAMA_DIR/src/llama-chat.cpp"                    "$CPP_DIR/llama-chat.cpp"
cp "$LLAMA_DIR/src/llama-context.h"                   "$CPP_DIR/llama-context.h"
cp "$LLAMA_DIR/src/llama-context.cpp"                 "$CPP_DIR/llama-context.cpp"
cp "$LLAMA_DIR/src/llama-mmap.h"                      "$CPP_DIR/llama-mmap.h"
cp "$LLAMA_DIR/src/llama-mmap.cpp"                    "$CPP_DIR/llama-mmap.cpp"
cp "$LLAMA_DIR/src/llama-model-loader.h"              "$CPP_DIR/llama-model-loader.h"
cp "$LLAMA_DIR/src/llama-model-loader.cpp"            "$CPP_DIR/llama-model-loader.cpp"
cp "$LLAMA_DIR/src/llama-model-saver.h"               "$CPP_DIR/llama-model-saver.h"
cp "$LLAMA_DIR/src/llama-model-saver.cpp"             "$CPP_DIR/llama-model-saver.cpp"
cp "$LLAMA_DIR/src/llama-model.h"                     "$CPP_DIR/llama-model.h"
cp "$LLAMA_DIR/src/llama-model.cpp"                   "$CPP_DIR/llama-model.cpp"
cp "$LLAMA_DIR/src/llama-kv-cells.h"                  "$CPP_DIR/llama-kv-cells.h"
cp "$LLAMA_DIR/src/llama-kv-cache.h"                  "$CPP_DIR/llama-kv-cache.h"
cp "$LLAMA_DIR/src/llama-kv-cache.cpp"                "$CPP_DIR/llama-kv-cache.cpp"
cp "$LLAMA_DIR/src/llama-kv-cache-dsa.h"              "$CPP_DIR/llama-kv-cache-dsa.h"
cp "$LLAMA_DIR/src/llama-kv-cache-dsa.cpp"            "$CPP_DIR/llama-kv-cache-dsa.cpp"
cp "$LLAMA_DIR/src/llama-kv-cache-iswa.h"             "$CPP_DIR/llama-kv-cache-iswa.h"
cp "$LLAMA_DIR/src/llama-kv-cache-iswa.cpp"           "$CPP_DIR/llama-kv-cache-iswa.cpp"
cp "$LLAMA_DIR/src/llama-memory-hybrid.h"             "$CPP_DIR/llama-memory-hybrid.h"
cp "$LLAMA_DIR/src/llama-memory-hybrid.cpp"           "$CPP_DIR/llama-memory-hybrid.cpp"
cp "$LLAMA_DIR/src/llama-memory-hybrid-iswa.h"        "$CPP_DIR/llama-memory-hybrid-iswa.h"
cp "$LLAMA_DIR/src/llama-memory-hybrid-iswa.cpp"      "$CPP_DIR/llama-memory-hybrid-iswa.cpp"
cp "$LLAMA_DIR/src/llama-memory-recurrent.h"          "$CPP_DIR/llama-memory-recurrent.h"
cp "$LLAMA_DIR/src/llama-memory-recurrent.cpp"        "$CPP_DIR/llama-memory-recurrent.cpp"
cp "$LLAMA_DIR/src/llama-adapter.h"                   "$CPP_DIR/llama-adapter.h"
cp "$LLAMA_DIR/src/llama-adapter.cpp"                 "$CPP_DIR/llama-adapter.cpp"
cp "$LLAMA_DIR/src/llama-arch.h"                      "$CPP_DIR/llama-arch.h"
cp "$LLAMA_DIR/src/llama-arch.cpp"                    "$CPP_DIR/llama-arch.cpp"
cp "$LLAMA_DIR/src/llama-batch.h"                     "$CPP_DIR/llama-batch.h"
cp "$LLAMA_DIR/src/llama-batch.cpp"                   "$CPP_DIR/llama-batch.cpp"
cp "$LLAMA_DIR/src/llama-cparams.h"                   "$CPP_DIR/llama-cparams.h"
cp "$LLAMA_DIR/src/llama-cparams.cpp"                 "$CPP_DIR/llama-cparams.cpp"
cp "$LLAMA_DIR/src/llama-hparams.h"                   "$CPP_DIR/llama-hparams.h"
cp "$LLAMA_DIR/src/llama-hparams.cpp"                 "$CPP_DIR/llama-hparams.cpp"
cp "$LLAMA_DIR/src/llama-impl.h"                      "$CPP_DIR/llama-impl.h"
cp "$LLAMA_DIR/src/llama-impl.cpp"                    "$CPP_DIR/llama-impl.cpp"
cp "$LLAMA_DIR/src/llama-ext.h"                       "$CPP_DIR/llama-ext.h"
cp "$LLAMA_DIR/src/llama-vocab.h"                     "$CPP_DIR/llama-vocab.h"
cp "$LLAMA_DIR/src/llama-vocab.cpp"                   "$CPP_DIR/llama-vocab.cpp"
cp "$LLAMA_DIR/src/llama-grammar.h"                   "$CPP_DIR/llama-grammar.h"
cp "$LLAMA_DIR/src/llama-grammar.cpp"                 "$CPP_DIR/llama-grammar.cpp"
cp "$LLAMA_DIR/src/llama-sampler.h"                   "$CPP_DIR/llama-sampler.h"
cp "$LLAMA_DIR/src/llama-sampler.cpp"                 "$CPP_DIR/llama-sampler.cpp"
cp "$LLAMA_DIR/src/unicode.h"                         "$CPP_DIR/unicode.h"
cp "$LLAMA_DIR/src/unicode.cpp"                       "$CPP_DIR/unicode.cpp"
cp "$LLAMA_DIR/src/unicode-data.h"                    "$CPP_DIR/unicode-data.h"
cp "$LLAMA_DIR/src/unicode-data.cpp"                  "$CPP_DIR/unicode-data.cpp"
cp "$LLAMA_DIR/src/llama-graph.h"                     "$CPP_DIR/llama-graph.h"
cp "$LLAMA_DIR/src/llama-graph.cpp"                   "$CPP_DIR/llama-graph.cpp"
cp "$LLAMA_DIR/src/llama-io.h"                        "$CPP_DIR/llama-io.h"
cp "$LLAMA_DIR/src/llama-io.cpp"                      "$CPP_DIR/llama-io.cpp"
cp "$LLAMA_DIR/src/llama-memory.h"                    "$CPP_DIR/llama-memory.h"
cp "$LLAMA_DIR/src/llama-memory.cpp"                  "$CPP_DIR/llama-memory.cpp"

# ---------------------------------------------------------------------------
# common
# ---------------------------------------------------------------------------
echo "==> Copying common..."
cp "$LLAMA_DIR/common/log.h"                          "$CPP_DIR/common/log.h"
cp "$LLAMA_DIR/common/log.cpp"                        "$CPP_DIR/common/log.cpp"
cp "$LLAMA_DIR/common/common.h"                       "$CPP_DIR/common/common.h"
cp "$LLAMA_DIR/common/common.cpp"                     "$CPP_DIR/common/common.cpp"
cp "$LLAMA_DIR/common/sampling.h"                     "$CPP_DIR/common/sampling.h"
cp "$LLAMA_DIR/common/sampling.cpp"                   "$CPP_DIR/common/sampling.cpp"
cp "$LLAMA_DIR/common/speculative.h"                  "$CPP_DIR/common/speculative.h"
cp "$LLAMA_DIR/common/speculative.cpp"                "$CPP_DIR/common/speculative.cpp"
cp "$LLAMA_DIR/common/ngram-cache.h"                  "$CPP_DIR/common/ngram-cache.h"
cp "$LLAMA_DIR/common/ngram-cache.cpp"                "$CPP_DIR/common/ngram-cache.cpp"
cp "$LLAMA_DIR/common/ngram-map.h"                    "$CPP_DIR/common/ngram-map.h"
cp "$LLAMA_DIR/common/ngram-map.cpp"                  "$CPP_DIR/common/ngram-map.cpp"
cp "$LLAMA_DIR/common/ngram-mod.h"                    "$CPP_DIR/common/ngram-mod.h"
cp "$LLAMA_DIR/common/ngram-mod.cpp"                  "$CPP_DIR/common/ngram-mod.cpp"
cp "$LLAMA_DIR/common/json-schema-to-grammar.h"       "$CPP_DIR/common/json-schema-to-grammar.h"
cp "$LLAMA_DIR/common/json-schema-to-grammar.cpp"     "$CPP_DIR/common/json-schema-to-grammar.cpp"
cp "$LLAMA_DIR/common/json-partial.h"                 "$CPP_DIR/common/json-partial.h"
cp "$LLAMA_DIR/common/json-partial.cpp"               "$CPP_DIR/common/json-partial.cpp"
cp "$LLAMA_DIR/common/regex-partial.h"                "$CPP_DIR/common/regex-partial.h"
cp "$LLAMA_DIR/common/regex-partial.cpp"              "$CPP_DIR/common/regex-partial.cpp"
cp "$LLAMA_DIR/common/chat.h"                         "$CPP_DIR/common/chat.h"
cp "$LLAMA_DIR/common/chat.cpp"                       "$CPP_DIR/common/chat.cpp"
cp "$LLAMA_DIR/common/chat-auto-parser.h"             "$CPP_DIR/common/chat-auto-parser.h"
cp "$LLAMA_DIR/common/chat-auto-parser-helpers.h"     "$CPP_DIR/common/chat-auto-parser-helpers.h"
cp "$LLAMA_DIR/common/chat-auto-parser-helpers.cpp"   "$CPP_DIR/common/chat-auto-parser-helpers.cpp"
cp "$LLAMA_DIR/common/chat-auto-parser-generator.cpp" "$CPP_DIR/common/chat-auto-parser-generator.cpp"
cp "$LLAMA_DIR/common/chat-diff-analyzer.cpp"         "$CPP_DIR/common/chat-diff-analyzer.cpp"
cp "$LLAMA_DIR/common/chat-peg-parser.h"              "$CPP_DIR/common/chat-peg-parser.h"
cp "$LLAMA_DIR/common/chat-peg-parser.cpp"            "$CPP_DIR/common/chat-peg-parser.cpp"
cp "$LLAMA_DIR/common/peg-parser.h"                   "$CPP_DIR/common/peg-parser.h"
cp "$LLAMA_DIR/common/peg-parser.cpp"                 "$CPP_DIR/common/peg-parser.cpp"
cp "$LLAMA_DIR/common/unicode.h"                      "$CPP_DIR/common/unicode.h"
cp "$LLAMA_DIR/common/unicode.cpp"                    "$CPP_DIR/common/unicode.cpp"
cp "$LLAMA_DIR/common/reasoning-budget.h"             "$CPP_DIR/common/reasoning-budget.h"
cp "$LLAMA_DIR/common/reasoning-budget.cpp"           "$CPP_DIR/common/reasoning-budget.cpp"
cp "$LLAMA_DIR/common/fit.h"                          "$CPP_DIR/common/fit.h"
cp "$LLAMA_DIR/common/fit.cpp"                        "$CPP_DIR/common/fit.cpp"
cp "$LLAMA_DIR/common/build-info.h"                   "$CPP_DIR/common/build-info.h"

# ---------------------------------------------------------------------------
# jinja (template engine, needed by chat.cpp)
# ---------------------------------------------------------------------------
echo "==> Copying jinja..."
rm -rf "$CPP_DIR/common/jinja"
cp -r "$LLAMA_DIR/common/jinja" "$CPP_DIR/common/jinja"
# Rename jinja/string.h to avoid conflict with system <string.h>
mv "$CPP_DIR/common/jinja/string.h" "$CPP_DIR/common/jinja/jinja-string.h"
sed -i 's|#include "string.h"|#include "jinja-string.h"|g' "$CPP_DIR/common/jinja/value.h"
sed -i 's|#include "jinja/string.h"|#include "jinja/jinja-string.h"|g' "$CPP_DIR/common/jinja/string.cpp"
sed -i 's|#include "../src/llama-ext.h"|#include "../llama-ext.h"|g' "$CPP_DIR/common/fit.h"
sed -i 's|#include "../src/llama-ext.h"|#include "../llama-ext.h"|g' "$CPP_DIR/common/fit.cpp"
sed -i 's|#include "../src/llama-ext.h"|#include "../llama-ext.h"|g' "$CPP_DIR/common/speculative.cpp"

# ---------------------------------------------------------------------------
# nlohmann + vendors
# ---------------------------------------------------------------------------
echo "==> Copying nlohmann + vendors..."
rm -rf "$CPP_DIR/nlohmann"
cp -r "$LLAMA_DIR/vendor/nlohmann" "$CPP_DIR/nlohmann"

# mtmd audio/image vendors
rm -rf "$CPP_DIR/tools/mtmd/miniaudio"
rm -rf "$CPP_DIR/tools/mtmd/stb"
cp -r "$LLAMA_DIR/vendor/miniaudio" "$CPP_DIR/tools/mtmd/miniaudio"
cp -r "$LLAMA_DIR/vendor/stb" "$CPP_DIR/tools/mtmd/stb"

# ---------------------------------------------------------------------------
# tools/mtmd (multimodal)
# ---------------------------------------------------------------------------
echo "==> Copying tools/mtmd..."
rm -rf "$CPP_DIR/tools/mtmd"/*.{h,cpp} 2>/dev/null || true
mkdir -p "$CPP_DIR/tools/mtmd"
cp "$LLAMA_DIR/tools/mtmd/mtmd.h"          "$CPP_DIR/tools/mtmd/mtmd.h"
cp "$LLAMA_DIR/tools/mtmd/mtmd.cpp"        "$CPP_DIR/tools/mtmd/mtmd.cpp"
cp "$LLAMA_DIR/tools/mtmd/clip.h"          "$CPP_DIR/tools/mtmd/clip.h"
cp "$LLAMA_DIR/tools/mtmd/clip.cpp"        "$CPP_DIR/tools/mtmd/clip.cpp"
cp "$LLAMA_DIR/tools/mtmd/clip-impl.h"     "$CPP_DIR/tools/mtmd/clip-impl.h"
cp "$LLAMA_DIR/tools/mtmd/clip-model.h"    "$CPP_DIR/tools/mtmd/clip-model.h"
cp "$LLAMA_DIR/tools/mtmd/clip-graph.h"    "$CPP_DIR/tools/mtmd/clip-graph.h"
cp "$LLAMA_DIR/tools/mtmd/mtmd-helper.cpp" "$CPP_DIR/tools/mtmd/mtmd-helper.cpp"
cp "$LLAMA_DIR/tools/mtmd/mtmd-helper.h"   "$CPP_DIR/tools/mtmd/mtmd-helper.h"
cp "$LLAMA_DIR/tools/mtmd/mtmd-audio.h"    "$CPP_DIR/tools/mtmd/mtmd-audio.h"
cp "$LLAMA_DIR/tools/mtmd/mtmd-audio.cpp"  "$CPP_DIR/tools/mtmd/mtmd-audio.cpp"
cp "$LLAMA_DIR/tools/mtmd/mtmd-image.h"    "$CPP_DIR/tools/mtmd/mtmd-image.h"
cp "$LLAMA_DIR/tools/mtmd/mtmd-image.cpp"  "$CPP_DIR/tools/mtmd/mtmd-image.cpp"
# mtmd models/debug subdirs — use /* to copy contents, not the dir itself
if [ -d "$LLAMA_DIR/tools/mtmd/models" ]; then
  mkdir -p "$CPP_DIR/tools/mtmd/models"
  cp -r "$LLAMA_DIR/tools/mtmd/models/"* "$CPP_DIR/tools/mtmd/models/" 2>/dev/null || true
fi
if [ -d "$LLAMA_DIR/tools/mtmd/debug" ]; then
  mkdir -p "$CPP_DIR/tools/mtmd/debug"
  cp -r "$LLAMA_DIR/tools/mtmd/debug/"* "$CPP_DIR/tools/mtmd/debug/" 2>/dev/null || true
fi
# Copy vendors into mtmd — contents only, mkdir first
mkdir -p "$CPP_DIR/tools/mtmd/miniaudio"
cp -r "$LLAMA_DIR/vendor/miniaudio/"* "$CPP_DIR/tools/mtmd/miniaudio/" 2>/dev/null || true
mkdir -p "$CPP_DIR/tools/mtmd/stb"
cp -r "$LLAMA_DIR/vendor/stb/"* "$CPP_DIR/tools/mtmd/stb/" 2>/dev/null || true

# ---------------------------------------------------------------------------
# rn-llama glue (GARDER: engine-facing files, zero JSI)
# ---------------------------------------------------------------------------
echo "==> Copying rn-llama glue from $LLAMA_RN_CPP..."
if [ ! -d "$LLAMA_RN_CPP" ]; then
  echo "ERROR: llama.rn cpp/ missing at $LLAMA_RN_CPP despite a verified checkout"
  exit 1
fi

cp "$LLAMA_RN_CPP/rn-llama.h"         "$CPP_DIR/rn-llama.h"
cp "$LLAMA_RN_CPP/rn-llama.cpp"       "$CPP_DIR/rn-llama.cpp"
cp "$LLAMA_RN_CPP/rn-completion.h"    "$CPP_DIR/rn-completion.h"
cp "$LLAMA_RN_CPP/rn-completion.cpp"  "$CPP_DIR/rn-completion.cpp"
cp "$LLAMA_RN_CPP/rn-slot.h"          "$CPP_DIR/rn-slot.h"
cp "$LLAMA_RN_CPP/rn-slot.cpp"        "$CPP_DIR/rn-slot.cpp"
cp "$LLAMA_RN_CPP/rn-slot-manager.h"  "$CPP_DIR/rn-slot-manager.h"
cp "$LLAMA_RN_CPP/rn-slot-manager.cpp" "$CPP_DIR/rn-slot-manager.cpp"
cp "$LLAMA_RN_CPP/rn-common.hpp"      "$CPP_DIR/rn-common.hpp"
cp "$LLAMA_RN_CPP/rn-mtmd.hpp"        "$CPP_DIR/rn-mtmd.hpp"
cp "$LLAMA_RN_CPP/rn-tts.h"           "$CPP_DIR/rn-tts.h"
cp "$LLAMA_RN_CPP/rn-tts.cpp"         "$CPP_DIR/rn-tts.cpp"

# jsi/ — only the two allowed files (JSINativeHeaders.h, ThreadPool.{h,cpp})
cp "$LLAMA_RN_CPP/jsi/JSINativeHeaders.h"  "$CPP_DIR/jsi/JSINativeHeaders.h"
cp "$LLAMA_RN_CPP/jsi/ThreadPool.h"        "$CPP_DIR/jsi/ThreadPool.h"
cp "$LLAMA_RN_CPP/jsi/ThreadPool.cpp"      "$CPP_DIR/jsi/ThreadPool.cpp"

# ---------------------------------------------------------------------------
# LM_ prefix rewrite on ggml/gguf symbols
# ---------------------------------------------------------------------------
echo "==> Applying LM_ prefix rewrites..."

files_add_lm_prefix=(
  "$CPP_DIR/ggml-opencl/"*.cpp

  # Vulkan backend (ggml-vulkan.cpp and its header)
  "$CPP_DIR/ggml-vulkan/ggml-vulkan.cpp"
  "$CPP_DIR/ggml-vulkan/ggml-vulkan.h"

  "$CPP_DIR/ggml-cpu/"*.h
  "$CPP_DIR/ggml-cpu/"*.c
  "$CPP_DIR/ggml-cpu/"*.cpp
  "$CPP_DIR/ggml-cpu/amx/"*.h
  "$CPP_DIR/ggml-cpu/amx/"*.cpp
  "$CPP_DIR/ggml-cpu/arch/arm/"*.c
  "$CPP_DIR/ggml-cpu/arch/arm/"*.cpp
  "$CPP_DIR/ggml-cpu/arch/x86/"*.c
  "$CPP_DIR/ggml-cpu/arch/x86/"*.cpp

  # Model definitions
  "$CPP_DIR/models/"*.h
  "$CPP_DIR/models/"*.cpp

  # Multimodal files
  "$CPP_DIR/tools/mtmd/"*.h
  "$CPP_DIR/tools/mtmd/"*.cpp
  "$CPP_DIR/tools/mtmd/models/"*.h
  "$CPP_DIR/tools/mtmd/models/"*.cpp
  "$CPP_DIR/tools/mtmd/debug/"*.h
  "$CPP_DIR/tools/mtmd/debug/"*.cpp

  # llama/ggml top-level
  "$CPP_DIR/"*.h
  "$CPP_DIR/"*.cpp
  "$CPP_DIR/"*.c

  "$CPP_DIR/common/"*.h
  "$CPP_DIR/common/"*.cpp
)

for file in "${files_add_lm_prefix[@]}"; do
  # Skip if glob didn't match
  [ -f "$file" ] || continue

  # Skip rn-* glue (they use lm_ already via the llama.rn vendored headers)
  [[ "$file" == *"/rn-"* ]] && continue

  # Skip the local ggml-ext.h shim if present
  [[ "$file" == "$CPP_DIR/ggml-ext.h" ]] && continue

  sed -i 's/GGML_/LM_GGML_/g' "$file"
  sed -i 's/ggml_/lm_ggml_/g' "$file"
  sed -i 's/GGUF_/LM_GGUF_/g' "$file"
  sed -i 's/gguf_/lm_gguf_/g' "$file"

  # nlohmann: use local relative include instead of system angle-bracket form
  sed -i 's|<nlohmann/json.hpp>|"nlohmann/json.hpp"|g' "$file"
  sed -i 's|<nlohmann/json_fwd.hpp>|"nlohmann/json_fwd.hpp"|g' "$file"
done

# iq prefix rewrites (avoid collision with other ggml users like whisper.rn)
files_iq=(
  "$CPP_DIR/ggml-quants.h"
  "$CPP_DIR/ggml-quants.c"
  "$CPP_DIR/ggml.c"
)
for file in "${files_iq[@]}"; do
  [ -f "$file" ] || continue
  sed -i 's/iq2xs_init_impl/lm_iq2xs_init_impl/g' "$file"
  sed -i 's/iq2xs_free_impl/lm_iq2xs_free_impl/g' "$file"
  sed -i 's/iq3xs_init_impl/lm_iq3xs_init_impl/g' "$file"
  sed -i 's/iq3xs_free_impl/lm_iq3xs_free_impl/g' "$file"
done

# ---------------------------------------------------------------------------
# Apply patches from llama.rn (Android-compatible ones)
# ---------------------------------------------------------------------------
echo "==> Applying patches inherited from llama.rn..."
if [ ! -d "$PATCHES_SRC" ]; then
  echo "ERROR: inherited patches not found at $PATCHES_SRC"
  echo "  The pinned llama.rn checkout is incomplete."
  exit 1
fi

# The only patches allowed not to apply are those whose target we deliberately
# don't vendor (see the ggml-metal / ggml-hexagon skips at the top). Everything
# else failing means the sources moved under the patch — same discipline as the
# local patches/ below, because a skipped hunk here is a fix silently lost.
patches_not_vendored=(
  ggml-metal-flash-attn-smem.patch   # ggml-metal/ggml-metal-ops.cpp — iOS only
  ggml-metal-mul-mv-id-tiitg.patch   # ggml-metal/ggml-metal.metal   — iOS only
  ggml-hexagon.cpp.patch             # ggml-hexagon/                 — not needed
)

for patch_file in "$PATCHES_SRC"/*.patch; do
  [ -f "$patch_file" ] || continue
  patch_name=$(basename "$patch_file")

  skip=0
  for not_vendored in "${patches_not_vendored[@]}"; do
    [ "$patch_name" = "$not_vendored" ] && skip=1
  done
  if [ "$skip" -eq 1 ]; then
    echo "  Skipping patch (target not vendored): $patch_name"
    continue
  fi

  echo "  Applying patch: $patch_name"
  # --batch: never prompt. A non-zero exit means at least one hunk was rejected.
  if ! patch --batch -p0 -d "$CPP_DIR" < "$patch_file"; then
    echo ""
    echo "  ERROR: inherited patch $patch_name does not apply cleanly."
    echo "  Either third_party/llama.cpp moved under it, or llama.rn changed it."
    echo "  Resolve deliberately — do not skip it: bump scripts/llama.rn.rev to a"
    echo "  llama.rn revision matching the llama.cpp pin, or add $patch_name to"
    echo "  patches_not_vendored above with the reason. See the .rej files in"
    echo "  $CPP_DIR."
    exit 1
  fi
done

# Cleanup .orig files
find "$CPP_DIR" -name "*.orig" -delete

# ---------------------------------------------------------------------------
# Apply local patches (ours, from patches/ at repo root)
# Unlike the llama.rn patches above, these MUST apply: they carry fixes the
# lib depends on (e.g. the Vulkan UMA descriptor fix, upstream #23057).
# If one fails after a submodule bump, update the patch deliberately —
# do not skip it.
# ---------------------------------------------------------------------------
echo "==> Applying local patches (patches/)..."
if [ -d "$ROOT_DIR/patches" ]; then
  for patch_file in "$ROOT_DIR/patches"/*.patch; do
    [ -f "$patch_file" ] || continue
    echo "  Applying local patch: $(basename $patch_file)"
    if git -C "$ROOT_DIR" apply --check "$patch_file" 2>/dev/null; then
      git -C "$ROOT_DIR" apply "$patch_file"
    elif git -C "$ROOT_DIR" apply --reverse --check "$patch_file" 2>/dev/null; then
      echo "  Already applied, skipping."
    else
      echo "  ERROR: local patch $(basename $patch_file) does not apply."
      echo "  The vendored sources changed (submodule bump?) — update the patch."
      exit 1
    fi
  done
fi

# ---------------------------------------------------------------------------
# Generate build-info.cpp
# ---------------------------------------------------------------------------
echo "==> Generating build-info.cpp..."
cd "$LLAMA_DIR"
BUILD_NUMBER=$(git rev-list --count HEAD)
BUILD_COMMIT=$(git rev-parse --short=7 HEAD)
cd "$ROOT_DIR"

sed -e "s|@LLAMA_BUILD_NUMBER@|$BUILD_NUMBER|g" \
    -e "s|@LLAMA_BUILD_COMMIT@|$BUILD_COMMIT|g" \
    -e "s|@BUILD_COMPILER@|unknown|g" \
    -e "s|@BUILD_TARGET@|android|g" \
    "$LLAMA_DIR/common/build-info.cpp.in" > "$CPP_DIR/common/build-info.cpp"

echo ""
echo "==> Bootstrap complete!"
echo "    Output: $CPP_DIR"
echo "    llama.cpp commit: $(git -C "$LLAMA_DIR" rev-parse HEAD)"
echo "    llama.rn commit:  $LLAMA_RN_REV"
echo "    Build number: $BUILD_NUMBER, commit: $BUILD_COMMIT"
