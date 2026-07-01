#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define LM_GGML_VK_NAME "Vulkan"
#define LM_GGML_VK_MAX_DEVICES 16

// backend API
LM_GGML_BACKEND_API lm_ggml_backend_t lm_ggml_backend_vk_init(size_t dev_num);

LM_GGML_BACKEND_API bool lm_ggml_backend_is_vk(lm_ggml_backend_t backend);
LM_GGML_BACKEND_API int  lm_ggml_backend_vk_get_device_count(void);
LM_GGML_BACKEND_API void lm_ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
LM_GGML_BACKEND_API void lm_ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

LM_GGML_BACKEND_API lm_ggml_backend_buffer_type_t lm_ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
LM_GGML_BACKEND_API lm_ggml_backend_buffer_type_t lm_ggml_backend_vk_host_buffer_type(void);

LM_GGML_BACKEND_API lm_ggml_backend_reg_t lm_ggml_backend_vk_reg(void);

#ifdef  __cplusplus
}
#endif
