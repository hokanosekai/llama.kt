cmake_minimum_required(VERSION 3.22)

project(llamakt)

find_package(Python3 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ── Vulkan (Android system library) ──────────────────────────────────────────
find_library(VULKAN_LIB vulkan)

# ── Include directories ─────────────────────────────────────────────────────
include_directories(
    ${SRC_DIR}
    ${SRC_DIR}/common
    ${SRC_DIR}/common/jinja
    ${SRC_DIR}/ggml-cpu
    ${SRC_DIR}/tools/mtmd
    ${SRC_DIR}/nlohmann
    ${SRC_DIR}/jsi
    # Vulkan backend sources + generated shader header
    ${SRC_DIR}/ggml-vulkan
    # SPIR-V headers (for spirv/unified1/spirv.hpp)
    ${SRC_DIR}/spirv-headers
)

# ── Source lists ─────────────────────────────────────────────────────────────

# Core GGML
set(GGML_CORE_SOURCES
    ${SRC_DIR}/ggml.c
    ${SRC_DIR}/ggml-alloc.c
    ${SRC_DIR}/ggml-backend.cpp
    ${SRC_DIR}/ggml-backend-dl.cpp
    ${SRC_DIR}/ggml-backend-meta.cpp
    ${SRC_DIR}/ggml-backend-reg.cpp
    ${SRC_DIR}/ggml-opt.cpp
    ${SRC_DIR}/ggml-threading.cpp
    ${SRC_DIR}/ggml-quants.c
    ${SRC_DIR}/gguf.cpp
)

# GGML CPU backend — explicit, arm64 only (exclude x86/)
file(GLOB GGML_CPU_C_FILES   CONFIGURE_DEPENDS ${SRC_DIR}/ggml-cpu/*.c)
file(GLOB GGML_CPU_CPP_FILES CONFIGURE_DEPENDS ${SRC_DIR}/ggml-cpu/*.cpp)
file(GLOB GGML_CPU_AMX_FILES CONFIGURE_DEPENDS ${SRC_DIR}/ggml-cpu/amx/*.cpp)

# ARM arch-specific files
set(GGML_CPU_ARCH_SOURCES
    ${SRC_DIR}/ggml-cpu/arch/arm/quants.c
    ${SRC_DIR}/ggml-cpu/arch/arm/repack.cpp
    ${SRC_DIR}/ggml-cpu/arch/arm/cpu-feats.cpp
)

# Llama core
file(GLOB LLAMA_FILES CONFIGURE_DEPENDS ${SRC_DIR}/llama*.cpp)
set(LLAMA_EXTRA_SOURCES
    ${SRC_DIR}/unicode-data.cpp
    ${SRC_DIR}/unicode.cpp
)

# Common utilities
file(GLOB COMMON_FILES CONFIGURE_DEPENDS ${SRC_DIR}/common/*.cpp)
file(GLOB JINJA_FILES  CONFIGURE_DEPENDS ${SRC_DIR}/common/jinja/*.cpp)

# Multimodal (mtmd)
file(GLOB MTMD_FILES       CONFIGURE_DEPENDS ${SRC_DIR}/tools/mtmd/*.cpp)
file(GLOB MTMD_MODEL_FILES CONFIGURE_DEPENDS ${SRC_DIR}/tools/mtmd/models/*.cpp)

# Model backends
file(GLOB MODEL_FILES CONFIGURE_DEPENDS ${SRC_DIR}/models/*.cpp)

# RN glue layer
set(RN_GLUE_SOURCES
    ${SRC_DIR}/rn-llama.cpp
    ${SRC_DIR}/rn-completion.cpp
    ${SRC_DIR}/rn-tts.cpp
    ${SRC_DIR}/rn-slot.cpp
    ${SRC_DIR}/rn-slot-manager.cpp
    ${SRC_DIR}/anyascii.c
)

# JSI helpers
set(JSI_SOURCES
    ${SRC_DIR}/jsi/ThreadPool.cpp
)

# JNI stub
set(JNI_STUB ${SRC_DIR}/tensai_jni.cpp)

# ── Vulkan backend + generated SPIR-V shaders ────────────────────────────────
set(GGML_VULKAN_SOURCES
    ${SRC_DIR}/ggml-vulkan/ggml-vulkan.cpp
)
file(GLOB GGML_VULKAN_SHADER_SOURCES CONFIGURE_DEPENDS
    ${SRC_DIR}/ggml-vulkan/shaders/*.cpp
)

# ── OpenCL kernel embedding ──────────────────────────────────────────────────
set(GGML_OPENCL_KERNELS
    add
    add_id
    argsort
    tri
    fill
    clamp
    cpy
    cvt
    diag_mask_inf
    diag
    div
    gelu
    get_rows
    glu
    group_norm
    solve_tri
    im2col_f32
    im2col_f16
    mean
    mul_mv_f16_f16
    mul_mv_f16_f32_1row
    mul_mv_f16_f32_l4
    mul_mv_f16_f32
    mul_mv_f32_f32
    mul_mv_q4_0_f32
    mul_mv_q4_0_f32_v
    mul_mv_q4_0_f32_8x_flat
    mul_mv_q4_0_f32_1d_8x_flat
    mul_mv_q4_0_f32_1d_16x_flat
    mul_mv_q4_1_f32
    mul_mv_q4_1_f32_flat
    mul_mv_q4_k_f32
    mul_mv_q4_k_f32_flat
    mul_mv_q5_0_f32
    mul_mv_q5_0_f32_flat
    mul_mv_q5_1_f32
    mul_mv_q5_1_f32_flat
    mul_mv_q5_k_f32
    mul_mv_q5_k_f32_flat
    mul_mv_q6_k_f32
    mul_mv_q6_k_f32_flat
    mul_mv_q8_0_f32
    mul_mv_q8_0_f32_flat
    mul_mv_iq4_nl_f32
    mul_mv_iq4_nl_f32_flat
    mul_mv_mxfp4_f32
    mul_mv_mxfp4_f32_flat
    mul_mv_id_q4_0_f32_8x_flat
    mul_mv_id_q8_0_f32
    mul_mv_id_q8_0_f32_flat
    mul_mv_id_mxfp4_f32
    mul_mv_id_mxfp4_f32_flat
    gemm_moe_q4_0_f32_ns
    gemv_moe_q4_0_f32_ns
    gemm_moe_q4_1_f32_ns
    gemv_moe_q4_1_f32_ns
    gemm_moe_q5_0_f32_ns
    gemv_moe_q5_0_f32_ns
    gemm_moe_q5_1_f32_ns
    gemv_moe_q5_1_f32_ns
    gemm_moe_q4_k_f32_ns
    gemv_moe_q4_k_f32_ns
    gemm_moe_q5_k_f32_ns
    gemv_moe_q5_k_f32_ns
    gemm_moe_q6_k_f32_ns
    gemv_moe_q6_k_f32_ns
    gemm_moe_mxfp4_f32
    gemv_moe_mxfp4_f32
    gemm_moe_mxfp4_f32_ns
    gemv_moe_mxfp4_f32_ns
    moe_reorder_b
    moe_sort_by_expert
    mul_mm_f32_f32_l4_lm
    mul_mm_f16_f32_l4_lm
    mul_mm_q4_0_f32_l4_lm
    mul_mm_q4_1_f32_l4_lm
    mul_mm_q5_0_f32_l4_lm
    mul_mm_q5_1_f32_l4_lm
    mul_mm_q8_0_f32_l4_lm
    mul_mm_iq4_nl_f32_l4_lm
    mul_mm_q4_k_f32_l4_lm
    mul_mm_q5_k_f32_l4_lm
    mul_mm_q6_k_f32_l4_lm
    gemv_noshuffle_q4_0_f32
    gemv_noshuffle_q4_0_f32_spec
    gemm_noshuffle_q4_0_f32
    gemv_noshuffle_q4_1_f32
    gemm_noshuffle_q4_1_f32
    gemv_noshuffle_iq4_nl_f32
    gemm_noshuffle_iq4_nl_f32
    gemv_noshuffle_q5_0_f32
    gemm_noshuffle_q5_0_f32
    gemv_noshuffle_q5_1_f32
    gemm_noshuffle_q5_1_f32
    gemv_noshuffle_q8_0_f32
    gemm_noshuffle_q8_0_f32
    gemv_noshuffle_q4_k_f32
    gemm_noshuffle_q4_k_f32
    gemv_noshuffle_q6_k_f32
    gemm_noshuffle_q6_k_f32
    gemv_noshuffle_q5_k_f32
    gemm_noshuffle_q5_k_f32
    mul
    neg
    norm
    relu
    l2_norm
    rms_norm
    rope
    scale
    set_rows
    sigmoid
    silu
    softmax_4_f32
    softmax_4_f16
    softmax_f32
    softmax_f16
    sqr
    sqrt
    ssm_conv
    gated_delta_net
    sub
    sum_rows
    cumsum
    transpose
    concat
    tsembd
    upscale
    tanh
    exp
    expm1
    softplus
    pad
    repeat
    mul_mat_f16_f32
    mul_mm_f16_f32_kq_kqv
    conv2d
    conv2d_f16_f32
    flash_attn_f32_f16
    flash_attn_f16
    flash_attn_f32
    gemm_xmem_f16_f32_os8
)

set(GGML_OPENCL_KERNEL_DIR ${SRC_DIR}/ggml-opencl/kernels)
set(GGML_OPENCL_KERNEL_HEADERS "")

foreach(kernel ${GGML_OPENCL_KERNELS})
    set(input_file  ${GGML_OPENCL_KERNEL_DIR}/${kernel}.cl)
    set(output_file ${CMAKE_CURRENT_BINARY_DIR}/${kernel}.cl.h)
    add_custom_command(
        OUTPUT  ${output_file}
        COMMAND ${Python3_EXECUTABLE}
                ${GGML_OPENCL_KERNEL_DIR}/embed_kernel.py
                ${input_file} ${output_file}
        DEPENDS ${input_file} ${GGML_OPENCL_KERNEL_DIR}/embed_kernel.py
        COMMENT "Embedding OpenCL kernel: ${kernel}.cl"
    )
    list(APPEND GGML_OPENCL_KERNEL_HEADERS ${output_file})
endforeach()

# ── Library target ────────────────────────────────────────────────────────────
add_library(llamakt SHARED
    ${GGML_CORE_SOURCES}
    ${GGML_CPU_C_FILES}
    ${GGML_CPU_CPP_FILES}
    ${GGML_CPU_AMX_FILES}
    ${GGML_CPU_ARCH_SOURCES}
    ${LLAMA_FILES}
    ${LLAMA_EXTRA_SOURCES}
    ${COMMON_FILES}
    ${JINJA_FILES}
    ${MTMD_MODEL_FILES}
    ${MTMD_FILES}
    ${MODEL_FILES}
    ${RN_GLUE_SOURCES}
    ${JSI_SOURCES}
    # OpenCL backend
    ${SRC_DIR}/ggml-opencl/ggml-opencl.cpp
    ${GGML_OPENCL_KERNEL_HEADERS}
    # Vulkan backend + SPIR-V shaders
    ${GGML_VULKAN_SOURCES}
    ${GGML_VULKAN_SHADER_SOURCES}
    # JNI stub
    ${JNI_STUB}
)

# ── Include dirs for the target ───────────────────────────────────────────────
target_include_directories(llamakt PRIVATE
    ${SRC_DIR}
    ${SRC_DIR}/common
    ${SRC_DIR}/common/jinja
    ${SRC_DIR}/ggml-cpu
    ${SRC_DIR}/tools/mtmd
    ${SRC_DIR}/nlohmann
    ${SRC_DIR}/jsi
    # OpenCL headers (third_party, relative to repo root)
    ${SRC_DIR}/../../../../third_party/OpenCL-Headers
    # Generated kernel headers
    ${CMAKE_CURRENT_BINARY_DIR}
    # Vulkan backend sources + generated shader header
    ${SRC_DIR}/ggml-vulkan
    # SPIR-V headers
    ${SRC_DIR}/spirv-headers
)


# ── Compile definitions ───────────────────────────────────────────────────────
target_compile_options(llamakt PRIVATE
    # Vulkan C++ headers must be found before NDK sysroot's older Vulkan headers
    # (NDK 27.2 ships VK_HEADER_VERSION 275; vulkan.hpp v1.4.350 requires 350)
    -I${SRC_DIR}/vulkan-hpp
    # CPU backend
    -DLM_GGML_USE_CPU
    -DLM_GGML_USE_CPU_REPACK
    # OpenCL backend (Adreno)
    -DLM_GGML_USE_OPENCL
    -DLM_GGML_OPENCL_USE_ADRENO_KERNELS
    -DLM_GGML_OPENCL_EMBED_KERNELS
    -DLM_GGML_OPENCL_SOA_Q
    # Vulkan backend (Mali-G68 + any Vulkan 1.1+ GPU)
    -DLM_GGML_USE_VULKAN
    # General
    -D_GNU_SOURCE
    -pthread
    # ARM64 target — armv8.2-a+dotprod only (no i8mm / no smmla)
    # i8mm (ARMv8.6) is NOT present on e.g. MediaTek Dimensity 900 (Cortex-A78, ARMv8.2)
    # → keeping +i8mm here causes SIGILL on those CPUs via lm_ggml_gemm_q8_0_4x8_q8_0
    -march=armv8.2-a+dotprod
    # Optimisation
    -O3 -DNDEBUG
    -ffunction-sections -fdata-sections
    -fvectorize
    -ffp-model=fast -fno-finite-math-only
    -flto
)

# ── Link libraries ────────────────────────────────────────────────────────────
find_library(LOG_LIB log)

# OpenCL stub (runtime link via jniLibs stub .so)
set(OPENCL_STUB ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libOpenCL.so)

target_link_directories(llamakt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}
)

target_link_libraries(llamakt PRIVATE
    ${LOG_LIB}
    android
    OpenCL
    # Vulkan — Android system library (present on API 24+, always available on Mali devices)
    ${VULKAN_LIB}
)

target_link_options(llamakt PRIVATE
    -Wl,--gc-sections
    -flto
)
