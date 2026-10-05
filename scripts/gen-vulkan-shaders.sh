#!/bin/bash
set -euo pipefail

# gen-vulkan-shaders.sh - generate the Vulkan SPIR-V shaders ggml-vulkan links against.
# Single copy of the procedure: bootstrap.sh, CI (native.yml) and the README call it.
#
# Usage: scripts/gen-vulkan-shaders.sh <path-to-glslc>
#
#   glslc   Compiler for the GLSL -> SPIR-V step. The only input that changes the
#           output: NDK glslc in CI, system glslc locally (`which glslc`).
#
# Reads    third_party/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/*.comp
# Writes   llama-kt/src/main/cpp/ggml-vulkan/ggml-vulkan-shaders.hpp
#          llama-kt/src/main/cpp/ggml-vulkan/shaders/*.comp.cpp
# Needs    cmake, g++, and the llama.cpp submodule checked out.
# Paths are relative to this script, so any cwd works. Intermediates go in a
# mktemp dir (honours TMPDIR) removed on exit.

if [ $# -ne 1 ]; then
  echo "Usage: $0 <path-to-glslc>" >&2
  exit 2
fi
GLSLC=$1
if [ ! -x "$GLSLC" ]; then
  echo "ERROR: glslc not executable: $GLSLC" >&2
  exit 1
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VK_SHADERS_SRC="$ROOT_DIR/third_party/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders"
VK_DEST_DIR="$ROOT_DIR/llama-kt/src/main/cpp/ggml-vulkan"

if [ ! -d "$VK_SHADERS_SRC" ]; then
  echo "ERROR: $VK_SHADERS_SRC missing (git submodule update --init third_party/llama.cpp)" >&2
  exit 1
fi

VK_WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/llama-kt-vk-shaders.XXXXXX")
trap 'rm -rf "$VK_WORK_DIR"' EXIT
VK_BUILD_DIR="$VK_WORK_DIR/build"
VK_SPV_DIR="$VK_WORK_DIR/spv"
VK_OUT_DIR="$VK_WORK_DIR/out"

# Start the outputs from empty: shaders upstream deletes would otherwise survive
# in shaders/, which CMakeLists globs into the build.
mkdir -p "$VK_BUILD_DIR" "$VK_SPV_DIR" "$VK_OUT_DIR" "$VK_DEST_DIR/shaders"
rm -f "$VK_DEST_DIR/shaders/"*.comp.cpp

# The variant set is fixed by vulkan-shaders-gen's CMakeLists: no
# GGML_VULKAN_*_GLSLC_SUPPORT is passed, so no host capability autodetection and
# the output depends only on (llama.cpp commit, glslc build).
echo "  Building vulkan-shaders-gen for host..."
cmake -S "$VK_SHADERS_SRC" -B "$VK_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -Wno-dev > /dev/null
cmake --build "$VK_BUILD_DIR" --config Release -j"$(nproc)" > /dev/null
VK_GEN="$VK_BUILD_DIR/vulkan-shaders-gen"

if [ ! -x "$VK_GEN" ]; then
  echo "ERROR: vulkan-shaders-gen build failed - cannot generate Vulkan shaders" >&2
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
    --glslc "$GLSLC" \
    --source "$comp" \
    --output-dir "$VK_SPV_DIR" \
    --target-hpp "$VK_OUT_DIR/ggml-vulkan-shaders.hpp" \
    --target-cpp "$VK_OUT_DIR/${base}.cpp"
done

cp "$VK_OUT_DIR/ggml-vulkan-shaders.hpp" "$VK_DEST_DIR/"

# Fix relative include in shader cpp files (they're compiled from shaders/ subdir)
for f in "$VK_OUT_DIR"/*.cpp; do
  sed 's|#include "ggml-vulkan-shaders.hpp"|#include "../ggml-vulkan-shaders.hpp"|g' "$f" \
    > "$VK_DEST_DIR/shaders/$(basename "$f")"
done

echo "  Generated $(find "$VK_DEST_DIR/shaders" -name '*.comp.cpp' | wc -l) shaders"
