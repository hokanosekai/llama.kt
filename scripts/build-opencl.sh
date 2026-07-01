#!/bin/bash -e

# build-opencl.sh — build libOpenCL.so from OpenCL-ICD-Loader for Android arm64-v8a
# Output: llama-kt/src/main/jniLibs/arm64-v8a/libOpenCL.so
# Adapted from mybigday/llama.rn scripts/build-opencl.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

OPENCL_ICD_SUBMODULE=third_party/OpenCL-ICD-Loader
OPENCL_HEADERS_SUBMODULE=third_party/OpenCL-Headers
OPENCL_HEADERS_DIR="$ROOT_DIR/$OPENCL_HEADERS_SUBMODULE"

git submodule update --init --recursive "$OPENCL_ICD_SUBMODULE"
git submodule update --init --recursive "$OPENCL_HEADERS_SUBMODULE"

# Use NDK 27.2.12479018 (installed on this machine)
NDK_VERSION=27.2.12479018
ANDROID_HOME="${ANDROID_HOME:-/opt/android-sdk}"
CMAKE_TOOLCHAIN_FILE=$ANDROID_HOME/ndk/$NDK_VERSION/build/cmake/android.toolchain.cmake
ANDROID_PLATFORM=android-21
CMAKE_BUILD_TYPE=Release

if [ ! -d "$ANDROID_HOME/ndk/$NDK_VERSION" ]; then
  echo "NDK $NDK_VERSION not found at $ANDROID_HOME/ndk/$NDK_VERSION"
  echo "Available NDK versions: $(ls $ANDROID_HOME/ndk 2>/dev/null || echo 'none')"
  exit 1
fi

CMAKE_PATH=$(which cmake)
if ! command -v "$CMAKE_PATH" &> /dev/null; then
  echo "cmake not found in PATH"
  exit 1
fi

n_cpu=$(nproc 2>/dev/null || echo 1)

OUTPUT_DIR="$ROOT_DIR/llama-kt/src/main/jniLibs/arm64-v8a"
mkdir -p "$OUTPUT_DIR"

t0=$(date +%s)
cd "$OPENCL_ICD_SUBMODULE"

build_opencl() {
  ABI=$1
  BUILD_DIR=build/$ABI

  rm -rf $BUILD_DIR
  mkdir -p $BUILD_DIR && cd $BUILD_DIR

  $CMAKE_PATH ../.. \
    -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
    -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE \
    -DANDROID_ABI=$ABI \
    -DANDROID_PLATFORM=$ANDROID_PLATFORM \
    -DANDROID_STL=c++_shared \
    -DOPENCL_ICD_LOADER_HEADERS_DIR=$OPENCL_HEADERS_DIR

  $CMAKE_PATH --build . --config Release -j $n_cpu

  cp libOpenCL.so "$OUTPUT_DIR/"
  cd ../..
  rm -rf $BUILD_DIR
}

build_opencl arm64-v8a

t1=$(date +%s)
echo "libOpenCL.so built successfully in $((t1 - t0)) seconds"
echo "Output: $OUTPUT_DIR/libOpenCL.so"
