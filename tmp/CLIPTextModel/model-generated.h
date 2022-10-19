
#pragma once
#include "logging.h"
#include "device_functions-generated.h"
#include "model_interface.h"
#include "raii_wrapper.h"
#include "macros.h"
#include <algorithm>
#include <deque>
#include <string>
#include <unordered_map>
#include <math.h>


void reshape_0(
  int64_t*,
  int64_t*,
  int64_t*
);

    void batch_gather_1(half* output,
                   const half* input,
                   const int64_t* indices,
                   const int64_t batch_num,
                   const int64_t indices_num,
                   const int64_t instance_size,
                   const int64_t gather_dim_size,
                   uint8_t* workspace,
                   cudaStream_t stream);
    

    void batch_gather_3(half* output,
                   const half* input,
                   const int64_t* indices,
                   const int64_t batch_num,
                   const int64_t indices_num,
                   const int64_t instance_size,
                   const int64_t gather_dim_size,
                   uint8_t* workspace,
                   cudaStream_t stream);
    


void invoke_fused_elementwise_176(half* output0, const half* input0,const half* input1,  int n_elements, cudaStream_t stream);
    

    cudaError_t layernorm_7(half* output,
                   half* input,
                   const half* gamma,
                   const half* beta,
                   int m,
                   int n,
                   const float eps,
                   cudaStream_t stream);
    

void gemm_rcr_bias_8(
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  uint8_t*,

    int,


  int64_t*,

  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,

  int64_t*,

  cudaStream_t
);

    void flash_attention_10(half* output,
                   const half* qkv,
                   const int* cu_seqlens,
                   float* softmax_lse,
                   float* o_tmp,
                   int batch_size,
                   int seq_len,
                   int num_heads,
                   int head_size,
                   float p_dropout,
                   float softmax_scale,
                   bool is_causal,
                   bool loop,
                   cudaStream_t stream);
    

void gemm_rcr_bias_add_12(
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,

  cutlass::half_t*,
  uint8_t*,

    int,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,

  cudaStream_t
);

void gemm_rcr_bias_15(
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  uint8_t*,

    int,


  int64_t*,

  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,

  int64_t*,

  cudaStream_t
);

void invoke_fused_elementwise_177(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void gemm_rcr_bias_add_19(
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,
  cutlass::half_t*,

  cutlass::half_t*,
  uint8_t*,

    int,


  int64_t*,

  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,


  int64_t*,

  int64_t*,

  int64_t*,

  cudaStream_t
);

void invoke_fused_elementwise_178(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_179(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_180(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_181(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_182(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_183(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_184(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_185(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_186(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_187(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_188(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

#define CHECK_VECTOR_ACCESS(vector, idx)                                  \
  if (idx >= vector.size()) {                                             \
    throw std::out_of_range(                                              \
        "[__func__]: index out of range, " #vector ".size()=" +           \
        std::to_string(vector.size()) + ", got " + std::to_string(idx));  \
  }

namespace ait {
namespace {
void DeviceCheckLastError(const char* file, int line) {
  auto device_error = GetLastError();
  if (device_error != GetDeviceSuccess()) {
    std::string msg = std::string("Got error: ") + GetLastErrorString() +
                      " enum: " + std::to_string(device_error) +
                      " at " + file + ": " + std::to_string(line);
    LOG(ERROR) << msg;
    throw std::runtime_error(msg);
  }
}
}

// Model is the class that actually performs inference. It owns memory for
// intermediate tensors and dynamic dimensions. Constants are owned by
// the model's owning container object, and input/output memory is owned
// by the user.
// Once an inference run has started, it is not safe to re-use the Model
// until the run has finished!
class Model {
  public:
  Model(
      size_t blob_size,
      size_t workspace_size,
      size_t num_inputs,
      size_t num_outputs,
      size_t num_unbound_constants,
      uint8_t* constants)
      : blob(RAII_DeviceMalloc(blob_size)),
        workspace(RAII_DeviceMalloc(workspace_size)),
        params(num_inputs + num_outputs + num_unbound_constants),
        num_inputs(num_inputs),
        constants(constants) {
      dmlc::InitLogging("aitemplate"); // TODO(xxx): render network name
      LOG(INFO) << "Init AITemplate Runtime.";
      global_workspace = static_cast<uint8_t*>(workspace.get()) + 0;
      unique_workspace = static_cast<uint8_t*>(workspace.get());
      DEVICE_CHECK(GetDevice(&device_idx))
      DEVICE_CHECK(CreateEvent(&run_finished));
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
      DEVICE_CHECK(cudaDeviceGetAttribute(
        &max_smem_size, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_idx));
#endif
      DEVICE_CHECK(GetDeviceProperties(&device_properties, device_idx));
      DEVICE_CHECK(StreamCreate(&graph_capture_stream, /*non_blocking=*/true));

       constant_name_to_ptr_["embeddings_token_embedding_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&embeddings_token_embedding_weight));
     constant_name_to_ptr_["embeddings_position_embedding_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&embeddings_position_embedding_weight));
     constant_name_to_ptr_["encoder_layers_0_layer_norm1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_0_layer_norm1_weight));
     constant_name_to_ptr_["encoder_layers_0_layer_norm1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_0_layer_norm1_bias));
     constant_name_to_ptr_["encoder_layers_0_self_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_0_self_attn_qkv_weight));
     constant_name_to_ptr_["encoder_layers_0_self_attn_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_0_self_attn_qkv_bias));
    encoder_layers_0_self_attn_cu_length = reinterpret_cast<decltype(encoder_layers_0_self_attn_cu_length)>(constants + 0);
     constant_name_to_ptr_["encoder_layers_0_self_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_0_self_attn_proj_weight));
     constant_name_to_ptr_["encoder_layers_0_self_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_0_self_attn_proj_bias));
     constant_name_to_ptr_["encoder_layers_0_layer_norm2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_0_layer_norm2_weight));
     constant_name_to_ptr_["encoder_layers_0_layer_norm2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_0_layer_norm2_bias));
     constant_name_to_ptr_["encoder_layers_0_mlp_fc1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_0_mlp_fc1_weight));
     constant_name_to_ptr_["encoder_layers_0_mlp_fc1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_0_mlp_fc1_bias));
     constant_name_to_ptr_["encoder_layers_0_mlp_fc2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_0_mlp_fc2_weight));
     constant_name_to_ptr_["encoder_layers_0_mlp_fc2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_0_mlp_fc2_bias));
     constant_name_to_ptr_["encoder_layers_1_layer_norm1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_1_layer_norm1_weight));
     constant_name_to_ptr_["encoder_layers_1_layer_norm1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_1_layer_norm1_bias));
     constant_name_to_ptr_["encoder_layers_1_self_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_1_self_attn_qkv_weight));
     constant_name_to_ptr_["encoder_layers_1_self_attn_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_1_self_attn_qkv_bias));
    encoder_layers_1_self_attn_cu_length = reinterpret_cast<decltype(encoder_layers_1_self_attn_cu_length)>(constants + 64);
     constant_name_to_ptr_["encoder_layers_1_self_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_1_self_attn_proj_weight));
     constant_name_to_ptr_["encoder_layers_1_self_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_1_self_attn_proj_bias));
     constant_name_to_ptr_["encoder_layers_1_layer_norm2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_1_layer_norm2_weight));
     constant_name_to_ptr_["encoder_layers_1_layer_norm2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_1_layer_norm2_bias));
     constant_name_to_ptr_["encoder_layers_1_mlp_fc1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_1_mlp_fc1_weight));
     constant_name_to_ptr_["encoder_layers_1_mlp_fc1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_1_mlp_fc1_bias));
     constant_name_to_ptr_["encoder_layers_1_mlp_fc2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_1_mlp_fc2_weight));
     constant_name_to_ptr_["encoder_layers_1_mlp_fc2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_1_mlp_fc2_bias));
     constant_name_to_ptr_["encoder_layers_2_layer_norm1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_2_layer_norm1_weight));
     constant_name_to_ptr_["encoder_layers_2_layer_norm1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_2_layer_norm1_bias));
     constant_name_to_ptr_["encoder_layers_2_self_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_2_self_attn_qkv_weight));
     constant_name_to_ptr_["encoder_layers_2_self_attn_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_2_self_attn_qkv_bias));
    encoder_layers_2_self_attn_cu_length = reinterpret_cast<decltype(encoder_layers_2_self_attn_cu_length)>(constants + 128);
     constant_name_to_ptr_["encoder_layers_2_self_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_2_self_attn_proj_weight));
     constant_name_to_ptr_["encoder_layers_2_self_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_2_self_attn_proj_bias));
     constant_name_to_ptr_["encoder_layers_2_layer_norm2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_2_layer_norm2_weight));
     constant_name_to_ptr_["encoder_layers_2_layer_norm2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_2_layer_norm2_bias));
     constant_name_to_ptr_["encoder_layers_2_mlp_fc1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_2_mlp_fc1_weight));
     constant_name_to_ptr_["encoder_layers_2_mlp_fc1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_2_mlp_fc1_bias));
     constant_name_to_ptr_["encoder_layers_2_mlp_fc2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_2_mlp_fc2_weight));
     constant_name_to_ptr_["encoder_layers_2_mlp_fc2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_2_mlp_fc2_bias));
     constant_name_to_ptr_["encoder_layers_3_layer_norm1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_3_layer_norm1_weight));
     constant_name_to_ptr_["encoder_layers_3_layer_norm1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_3_layer_norm1_bias));
     constant_name_to_ptr_["encoder_layers_3_self_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_3_self_attn_qkv_weight));
     constant_name_to_ptr_["encoder_layers_3_self_attn_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_3_self_attn_qkv_bias));
    encoder_layers_3_self_attn_cu_length = reinterpret_cast<decltype(encoder_layers_3_self_attn_cu_length)>(constants + 192);
     constant_name_to_ptr_["encoder_layers_3_self_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_3_self_attn_proj_weight));
     constant_name_to_ptr_["encoder_layers_3_self_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_3_self_attn_proj_bias));
     constant_name_to_ptr_["encoder_layers_3_layer_norm2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_3_layer_norm2_weight));
     constant_name_to_ptr_["encoder_layers_3_layer_norm2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_3_layer_norm2_bias));
     constant_name_to_ptr_["encoder_layers_3_mlp_fc1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_3_mlp_fc1_weight));
     constant_name_to_ptr_["encoder_layers_3_mlp_fc1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_3_mlp_fc1_bias));
     constant_name_to_ptr_["encoder_layers_3_mlp_fc2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_3_mlp_fc2_weight));
     constant_name_to_ptr_["encoder_layers_3_mlp_fc2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_3_mlp_fc2_bias));
     constant_name_to_ptr_["encoder_layers_4_layer_norm1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_4_layer_norm1_weight));
     constant_name_to_ptr_["encoder_layers_4_layer_norm1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_4_layer_norm1_bias));
     constant_name_to_ptr_["encoder_layers_4_self_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_4_self_attn_qkv_weight));
     constant_name_to_ptr_["encoder_layers_4_self_attn_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_4_self_attn_qkv_bias));
    encoder_layers_4_self_attn_cu_length = reinterpret_cast<decltype(encoder_layers_4_self_attn_cu_length)>(constants + 256);
     constant_name_to_ptr_["encoder_layers_4_self_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_4_self_attn_proj_weight));
     constant_name_to_ptr_["encoder_layers_4_self_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_4_self_attn_proj_bias));
     constant_name_to_ptr_["encoder_layers_4_layer_norm2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_4_layer_norm2_weight));
     constant_name_to_ptr_["encoder_layers_4_layer_norm2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_4_layer_norm2_bias));
     constant_name_to_ptr_["encoder_layers_4_mlp_fc1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_4_mlp_fc1_weight));
     constant_name_to_ptr_["encoder_layers_4_mlp_fc1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_4_mlp_fc1_bias));
     constant_name_to_ptr_["encoder_layers_4_mlp_fc2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_4_mlp_fc2_weight));
     constant_name_to_ptr_["encoder_layers_4_mlp_fc2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_4_mlp_fc2_bias));
     constant_name_to_ptr_["encoder_layers_5_layer_norm1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_5_layer_norm1_weight));
     constant_name_to_ptr_["encoder_layers_5_layer_norm1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_5_layer_norm1_bias));
     constant_name_to_ptr_["encoder_layers_5_self_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_5_self_attn_qkv_weight));
     constant_name_to_ptr_["encoder_layers_5_self_attn_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_5_self_attn_qkv_bias));
    encoder_layers_5_self_attn_cu_length = reinterpret_cast<decltype(encoder_layers_5_self_attn_cu_length)>(constants + 320);
     constant_name_to_ptr_["encoder_layers_5_self_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_5_self_attn_proj_weight));
     constant_name_to_ptr_["encoder_layers_5_self_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_5_self_attn_proj_bias));
     constant_name_to_ptr_["encoder_layers_5_layer_norm2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_5_layer_norm2_weight));
     constant_name_to_ptr_["encoder_layers_5_layer_norm2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_5_layer_norm2_bias));
     constant_name_to_ptr_["encoder_layers_5_mlp_fc1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_5_mlp_fc1_weight));
     constant_name_to_ptr_["encoder_layers_5_mlp_fc1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_5_mlp_fc1_bias));
     constant_name_to_ptr_["encoder_layers_5_mlp_fc2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_5_mlp_fc2_weight));
     constant_name_to_ptr_["encoder_layers_5_mlp_fc2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_5_mlp_fc2_bias));
     constant_name_to_ptr_["encoder_layers_6_layer_norm1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_6_layer_norm1_weight));
     constant_name_to_ptr_["encoder_layers_6_layer_norm1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_6_layer_norm1_bias));
     constant_name_to_ptr_["encoder_layers_6_self_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_6_self_attn_qkv_weight));
     constant_name_to_ptr_["encoder_layers_6_self_attn_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_6_self_attn_qkv_bias));
    encoder_layers_6_self_attn_cu_length = reinterpret_cast<decltype(encoder_layers_6_self_attn_cu_length)>(constants + 384);
     constant_name_to_ptr_["encoder_layers_6_self_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_6_self_attn_proj_weight));
     constant_name_to_ptr_["encoder_layers_6_self_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_6_self_attn_proj_bias));
     constant_name_to_ptr_["encoder_layers_6_layer_norm2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_6_layer_norm2_weight));
     constant_name_to_ptr_["encoder_layers_6_layer_norm2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_6_layer_norm2_bias));
     constant_name_to_ptr_["encoder_layers_6_mlp_fc1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_6_mlp_fc1_weight));
     constant_name_to_ptr_["encoder_layers_6_mlp_fc1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_6_mlp_fc1_bias));
     constant_name_to_ptr_["encoder_layers_6_mlp_fc2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_6_mlp_fc2_weight));
     constant_name_to_ptr_["encoder_layers_6_mlp_fc2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_6_mlp_fc2_bias));
     constant_name_to_ptr_["encoder_layers_7_layer_norm1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_7_layer_norm1_weight));
     constant_name_to_ptr_["encoder_layers_7_layer_norm1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_7_layer_norm1_bias));
     constant_name_to_ptr_["encoder_layers_7_self_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_7_self_attn_qkv_weight));
     constant_name_to_ptr_["encoder_layers_7_self_attn_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_7_self_attn_qkv_bias));
    encoder_layers_7_self_attn_cu_length = reinterpret_cast<decltype(encoder_layers_7_self_attn_cu_length)>(constants + 448);
     constant_name_to_ptr_["encoder_layers_7_self_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_7_self_attn_proj_weight));
     constant_name_to_ptr_["encoder_layers_7_self_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_7_self_attn_proj_bias));
     constant_name_to_ptr_["encoder_layers_7_layer_norm2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_7_layer_norm2_weight));
     constant_name_to_ptr_["encoder_layers_7_layer_norm2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_7_layer_norm2_bias));
     constant_name_to_ptr_["encoder_layers_7_mlp_fc1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_7_mlp_fc1_weight));
     constant_name_to_ptr_["encoder_layers_7_mlp_fc1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_7_mlp_fc1_bias));
     constant_name_to_ptr_["encoder_layers_7_mlp_fc2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_7_mlp_fc2_weight));
     constant_name_to_ptr_["encoder_layers_7_mlp_fc2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_7_mlp_fc2_bias));
     constant_name_to_ptr_["encoder_layers_8_layer_norm1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_8_layer_norm1_weight));
     constant_name_to_ptr_["encoder_layers_8_layer_norm1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_8_layer_norm1_bias));
     constant_name_to_ptr_["encoder_layers_8_self_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_8_self_attn_qkv_weight));
     constant_name_to_ptr_["encoder_layers_8_self_attn_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_8_self_attn_qkv_bias));
    encoder_layers_8_self_attn_cu_length = reinterpret_cast<decltype(encoder_layers_8_self_attn_cu_length)>(constants + 512);
     constant_name_to_ptr_["encoder_layers_8_self_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_8_self_attn_proj_weight));
     constant_name_to_ptr_["encoder_layers_8_self_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_8_self_attn_proj_bias));
     constant_name_to_ptr_["encoder_layers_8_layer_norm2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_8_layer_norm2_weight));
     constant_name_to_ptr_["encoder_layers_8_layer_norm2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_8_layer_norm2_bias));
     constant_name_to_ptr_["encoder_layers_8_mlp_fc1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_8_mlp_fc1_weight));
     constant_name_to_ptr_["encoder_layers_8_mlp_fc1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_8_mlp_fc1_bias));
     constant_name_to_ptr_["encoder_layers_8_mlp_fc2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_8_mlp_fc2_weight));
     constant_name_to_ptr_["encoder_layers_8_mlp_fc2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_8_mlp_fc2_bias));
     constant_name_to_ptr_["encoder_layers_9_layer_norm1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_9_layer_norm1_weight));
     constant_name_to_ptr_["encoder_layers_9_layer_norm1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_9_layer_norm1_bias));
     constant_name_to_ptr_["encoder_layers_9_self_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_9_self_attn_qkv_weight));
     constant_name_to_ptr_["encoder_layers_9_self_attn_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_9_self_attn_qkv_bias));
    encoder_layers_9_self_attn_cu_length = reinterpret_cast<decltype(encoder_layers_9_self_attn_cu_length)>(constants + 576);
     constant_name_to_ptr_["encoder_layers_9_self_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_9_self_attn_proj_weight));
     constant_name_to_ptr_["encoder_layers_9_self_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_9_self_attn_proj_bias));
     constant_name_to_ptr_["encoder_layers_9_layer_norm2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_9_layer_norm2_weight));
     constant_name_to_ptr_["encoder_layers_9_layer_norm2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_9_layer_norm2_bias));
     constant_name_to_ptr_["encoder_layers_9_mlp_fc1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_9_mlp_fc1_weight));
     constant_name_to_ptr_["encoder_layers_9_mlp_fc1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_9_mlp_fc1_bias));
     constant_name_to_ptr_["encoder_layers_9_mlp_fc2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_9_mlp_fc2_weight));
     constant_name_to_ptr_["encoder_layers_9_mlp_fc2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_9_mlp_fc2_bias));
     constant_name_to_ptr_["encoder_layers_10_layer_norm1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_10_layer_norm1_weight));
     constant_name_to_ptr_["encoder_layers_10_layer_norm1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_10_layer_norm1_bias));
     constant_name_to_ptr_["encoder_layers_10_self_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_10_self_attn_qkv_weight));
     constant_name_to_ptr_["encoder_layers_10_self_attn_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_10_self_attn_qkv_bias));
    encoder_layers_10_self_attn_cu_length = reinterpret_cast<decltype(encoder_layers_10_self_attn_cu_length)>(constants + 640);
     constant_name_to_ptr_["encoder_layers_10_self_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_10_self_attn_proj_weight));
     constant_name_to_ptr_["encoder_layers_10_self_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_10_self_attn_proj_bias));
     constant_name_to_ptr_["encoder_layers_10_layer_norm2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_10_layer_norm2_weight));
     constant_name_to_ptr_["encoder_layers_10_layer_norm2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_10_layer_norm2_bias));
     constant_name_to_ptr_["encoder_layers_10_mlp_fc1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_10_mlp_fc1_weight));
     constant_name_to_ptr_["encoder_layers_10_mlp_fc1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_10_mlp_fc1_bias));
     constant_name_to_ptr_["encoder_layers_10_mlp_fc2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_10_mlp_fc2_weight));
     constant_name_to_ptr_["encoder_layers_10_mlp_fc2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_10_mlp_fc2_bias));
     constant_name_to_ptr_["encoder_layers_11_layer_norm1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_11_layer_norm1_weight));
     constant_name_to_ptr_["encoder_layers_11_layer_norm1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_11_layer_norm1_bias));
     constant_name_to_ptr_["encoder_layers_11_self_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_11_self_attn_qkv_weight));
     constant_name_to_ptr_["encoder_layers_11_self_attn_qkv_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_11_self_attn_qkv_bias));
    encoder_layers_11_self_attn_cu_length = reinterpret_cast<decltype(encoder_layers_11_self_attn_cu_length)>(constants + 704);
     constant_name_to_ptr_["encoder_layers_11_self_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_11_self_attn_proj_weight));
     constant_name_to_ptr_["encoder_layers_11_self_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_11_self_attn_proj_bias));
     constant_name_to_ptr_["encoder_layers_11_layer_norm2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_11_layer_norm2_weight));
     constant_name_to_ptr_["encoder_layers_11_layer_norm2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_11_layer_norm2_bias));
     constant_name_to_ptr_["encoder_layers_11_mlp_fc1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_11_mlp_fc1_weight));
     constant_name_to_ptr_["encoder_layers_11_mlp_fc1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_11_mlp_fc1_bias));
     constant_name_to_ptr_["encoder_layers_11_mlp_fc2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_11_mlp_fc2_weight));
     constant_name_to_ptr_["encoder_layers_11_mlp_fc2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&encoder_layers_11_mlp_fc2_bias));
     constant_name_to_ptr_["final_layer_norm_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&final_layer_norm_weight));
     constant_name_to_ptr_["final_layer_norm_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&final_layer_norm_bias));
      auto* blob_ptr = static_cast<uint8_t*>(blob.get());
      batch_gather_1_0 = reinterpret_cast<decltype(batch_gather_1_0)>(blob_ptr + 0);
    batch_gather_3_0 = reinterpret_cast<decltype(batch_gather_3_0)>(blob_ptr + 98304);
    size_5_0 = reinterpret_cast<decltype(size_5_0)>(blob_ptr + 197120);
    size_5_1 = reinterpret_cast<decltype(size_5_1)>(blob_ptr + 197184);
    reshape_6_0 = reinterpret_cast<decltype(reshape_6_0)>(blob_ptr + 294912);
    layernorm_7_0 = reinterpret_cast<decltype(layernorm_7_0)>(blob_ptr + 393216);
    reshape_9_0 = reinterpret_cast<decltype(reshape_9_0)>(blob_ptr + 0);
    flash_attention_10_0 = reinterpret_cast<decltype(flash_attention_10_0)>(blob_ptr + 393216);
    reshape_13_0 = reinterpret_cast<decltype(reshape_13_0)>(blob_ptr + 786432);
    layernorm_14_0 = reinterpret_cast<decltype(layernorm_14_0)>(blob_ptr + 393216);
    gemm_rcr_bias_15_0 = reinterpret_cast<decltype(gemm_rcr_bias_15_0)>(blob_ptr + 0);
    elementwise_18_0 = reinterpret_cast<decltype(elementwise_18_0)>(blob_ptr + 393216);
    reshape_20_0 = reinterpret_cast<decltype(reshape_20_0)>(blob_ptr + 294912);
    layernorm_21_0 = reinterpret_cast<decltype(layernorm_21_0)>(blob_ptr + 393216);
    reshape_23_0 = reinterpret_cast<decltype(reshape_23_0)>(blob_ptr + 0);
    flash_attention_24_0 = reinterpret_cast<decltype(flash_attention_24_0)>(blob_ptr + 393216);
    reshape_27_0 = reinterpret_cast<decltype(reshape_27_0)>(blob_ptr + 786432);
    layernorm_28_0 = reinterpret_cast<decltype(layernorm_28_0)>(blob_ptr + 393216);
    gemm_rcr_bias_29_0 = reinterpret_cast<decltype(gemm_rcr_bias_29_0)>(blob_ptr + 0);
    elementwise_32_0 = reinterpret_cast<decltype(elementwise_32_0)>(blob_ptr + 393216);
    reshape_34_0 = reinterpret_cast<decltype(reshape_34_0)>(blob_ptr + 294912);
    layernorm_35_0 = reinterpret_cast<decltype(layernorm_35_0)>(blob_ptr + 393216);
    reshape_37_0 = reinterpret_cast<decltype(reshape_37_0)>(blob_ptr + 0);
    flash_attention_38_0 = reinterpret_cast<decltype(flash_attention_38_0)>(blob_ptr + 393216);
    reshape_41_0 = reinterpret_cast<decltype(reshape_41_0)>(blob_ptr + 786432);
    layernorm_42_0 = reinterpret_cast<decltype(layernorm_42_0)>(blob_ptr + 393216);
    gemm_rcr_bias_43_0 = reinterpret_cast<decltype(gemm_rcr_bias_43_0)>(blob_ptr + 0);
    elementwise_46_0 = reinterpret_cast<decltype(elementwise_46_0)>(blob_ptr + 393216);
    reshape_48_0 = reinterpret_cast<decltype(reshape_48_0)>(blob_ptr + 294912);
    layernorm_49_0 = reinterpret_cast<decltype(layernorm_49_0)>(blob_ptr + 393216);
    reshape_51_0 = reinterpret_cast<decltype(reshape_51_0)>(blob_ptr + 0);
    flash_attention_52_0 = reinterpret_cast<decltype(flash_attention_52_0)>(blob_ptr + 393216);
    reshape_55_0 = reinterpret_cast<decltype(reshape_55_0)>(blob_ptr + 786432);
    layernorm_56_0 = reinterpret_cast<decltype(layernorm_56_0)>(blob_ptr + 393216);
    gemm_rcr_bias_57_0 = reinterpret_cast<decltype(gemm_rcr_bias_57_0)>(blob_ptr + 0);
    elementwise_60_0 = reinterpret_cast<decltype(elementwise_60_0)>(blob_ptr + 393216);
    reshape_62_0 = reinterpret_cast<decltype(reshape_62_0)>(blob_ptr + 294912);
    layernorm_63_0 = reinterpret_cast<decltype(layernorm_63_0)>(blob_ptr + 393216);
    reshape_65_0 = reinterpret_cast<decltype(reshape_65_0)>(blob_ptr + 0);
    flash_attention_66_0 = reinterpret_cast<decltype(flash_attention_66_0)>(blob_ptr + 393216);
    reshape_69_0 = reinterpret_cast<decltype(reshape_69_0)>(blob_ptr + 786432);
    layernorm_70_0 = reinterpret_cast<decltype(layernorm_70_0)>(blob_ptr + 393216);
    gemm_rcr_bias_71_0 = reinterpret_cast<decltype(gemm_rcr_bias_71_0)>(blob_ptr + 0);
    elementwise_74_0 = reinterpret_cast<decltype(elementwise_74_0)>(blob_ptr + 393216);
    reshape_76_0 = reinterpret_cast<decltype(reshape_76_0)>(blob_ptr + 294912);
    layernorm_77_0 = reinterpret_cast<decltype(layernorm_77_0)>(blob_ptr + 393216);
    reshape_79_0 = reinterpret_cast<decltype(reshape_79_0)>(blob_ptr + 0);
    flash_attention_80_0 = reinterpret_cast<decltype(flash_attention_80_0)>(blob_ptr + 393216);
    reshape_83_0 = reinterpret_cast<decltype(reshape_83_0)>(blob_ptr + 786432);
    layernorm_84_0 = reinterpret_cast<decltype(layernorm_84_0)>(blob_ptr + 393216);
    gemm_rcr_bias_85_0 = reinterpret_cast<decltype(gemm_rcr_bias_85_0)>(blob_ptr + 0);
    elementwise_88_0 = reinterpret_cast<decltype(elementwise_88_0)>(blob_ptr + 393216);
    reshape_90_0 = reinterpret_cast<decltype(reshape_90_0)>(blob_ptr + 294912);
    layernorm_91_0 = reinterpret_cast<decltype(layernorm_91_0)>(blob_ptr + 393216);
    reshape_93_0 = reinterpret_cast<decltype(reshape_93_0)>(blob_ptr + 0);
    flash_attention_94_0 = reinterpret_cast<decltype(flash_attention_94_0)>(blob_ptr + 393216);
    reshape_97_0 = reinterpret_cast<decltype(reshape_97_0)>(blob_ptr + 786432);
    layernorm_98_0 = reinterpret_cast<decltype(layernorm_98_0)>(blob_ptr + 393216);
    gemm_rcr_bias_99_0 = reinterpret_cast<decltype(gemm_rcr_bias_99_0)>(blob_ptr + 0);
    elementwise_102_0 = reinterpret_cast<decltype(elementwise_102_0)>(blob_ptr + 393216);
    reshape_104_0 = reinterpret_cast<decltype(reshape_104_0)>(blob_ptr + 294912);
    layernorm_105_0 = reinterpret_cast<decltype(layernorm_105_0)>(blob_ptr + 393216);
    reshape_107_0 = reinterpret_cast<decltype(reshape_107_0)>(blob_ptr + 0);
    flash_attention_108_0 = reinterpret_cast<decltype(flash_attention_108_0)>(blob_ptr + 393216);
    reshape_111_0 = reinterpret_cast<decltype(reshape_111_0)>(blob_ptr + 786432);
    layernorm_112_0 = reinterpret_cast<decltype(layernorm_112_0)>(blob_ptr + 393216);
    gemm_rcr_bias_113_0 = reinterpret_cast<decltype(gemm_rcr_bias_113_0)>(blob_ptr + 0);
    elementwise_116_0 = reinterpret_cast<decltype(elementwise_116_0)>(blob_ptr + 393216);
    reshape_118_0 = reinterpret_cast<decltype(reshape_118_0)>(blob_ptr + 294912);
    layernorm_119_0 = reinterpret_cast<decltype(layernorm_119_0)>(blob_ptr + 393216);
    reshape_121_0 = reinterpret_cast<decltype(reshape_121_0)>(blob_ptr + 0);
    flash_attention_122_0 = reinterpret_cast<decltype(flash_attention_122_0)>(blob_ptr + 393216);
    reshape_125_0 = reinterpret_cast<decltype(reshape_125_0)>(blob_ptr + 786432);
    layernorm_126_0 = reinterpret_cast<decltype(layernorm_126_0)>(blob_ptr + 393216);
    gemm_rcr_bias_127_0 = reinterpret_cast<decltype(gemm_rcr_bias_127_0)>(blob_ptr + 0);
    elementwise_130_0 = reinterpret_cast<decltype(elementwise_130_0)>(blob_ptr + 393216);
    reshape_132_0 = reinterpret_cast<decltype(reshape_132_0)>(blob_ptr + 294912);
    layernorm_133_0 = reinterpret_cast<decltype(layernorm_133_0)>(blob_ptr + 393216);
    reshape_135_0 = reinterpret_cast<decltype(reshape_135_0)>(blob_ptr + 0);
    flash_attention_136_0 = reinterpret_cast<decltype(flash_attention_136_0)>(blob_ptr + 393216);
    reshape_139_0 = reinterpret_cast<decltype(reshape_139_0)>(blob_ptr + 786432);
    layernorm_140_0 = reinterpret_cast<decltype(layernorm_140_0)>(blob_ptr + 393216);
    gemm_rcr_bias_141_0 = reinterpret_cast<decltype(gemm_rcr_bias_141_0)>(blob_ptr + 0);
    elementwise_144_0 = reinterpret_cast<decltype(elementwise_144_0)>(blob_ptr + 393216);
    reshape_146_0 = reinterpret_cast<decltype(reshape_146_0)>(blob_ptr + 294912);
    layernorm_147_0 = reinterpret_cast<decltype(layernorm_147_0)>(blob_ptr + 393216);
    reshape_149_0 = reinterpret_cast<decltype(reshape_149_0)>(blob_ptr + 0);
    flash_attention_150_0 = reinterpret_cast<decltype(flash_attention_150_0)>(blob_ptr + 393216);
    reshape_153_0 = reinterpret_cast<decltype(reshape_153_0)>(blob_ptr + 786432);
    layernorm_154_0 = reinterpret_cast<decltype(layernorm_154_0)>(blob_ptr + 393216);
    gemm_rcr_bias_155_0 = reinterpret_cast<decltype(gemm_rcr_bias_155_0)>(blob_ptr + 0);
    elementwise_158_0 = reinterpret_cast<decltype(elementwise_158_0)>(blob_ptr + 393216);
    reshape_160_0 = reinterpret_cast<decltype(reshape_160_0)>(blob_ptr + 294912);
    layernorm_161_0 = reinterpret_cast<decltype(layernorm_161_0)>(blob_ptr + 393216);
    reshape_163_0 = reinterpret_cast<decltype(reshape_163_0)>(blob_ptr + 0);
    flash_attention_164_0 = reinterpret_cast<decltype(flash_attention_164_0)>(blob_ptr + 393216);
    reshape_167_0 = reinterpret_cast<decltype(reshape_167_0)>(blob_ptr + 786432);
    layernorm_168_0 = reinterpret_cast<decltype(layernorm_168_0)>(blob_ptr + 393216);
    gemm_rcr_bias_169_0 = reinterpret_cast<decltype(gemm_rcr_bias_169_0)>(blob_ptr + 0);
    elementwise_172_0 = reinterpret_cast<decltype(elementwise_172_0)>(blob_ptr + 393216);
    reshape_174_0 = reinterpret_cast<decltype(reshape_174_0)>(blob_ptr + 0);
  
       params[0].shape_ptrs = {ParamDim(1, 1, &input0_dim_0), ParamDim(64, 64, &input0_dim_1)};
     params[3].shape_ptrs = {ParamDim(49408, 49408, &embeddings_token_embedding_weight_dim_0), ParamDim(768, 768, &embeddings_token_embedding_weight_dim_1)};
     params[1].shape_ptrs = {ParamDim(1, 1, &input1_dim_0), ParamDim(64, 64, &input1_dim_1)};
     params[4].shape_ptrs = {ParamDim(77, 77, &embeddings_position_embedding_weight_dim_0), ParamDim(768, 768, &embeddings_position_embedding_weight_dim_1)};
     params[5].shape_ptrs = {ParamDim(768, 768, &encoder_layers_0_layer_norm1_weight_dim_0)};
     params[6].shape_ptrs = {ParamDim(768, 768, &encoder_layers_0_layer_norm1_bias_dim_0)};
     params[7].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_0_self_attn_qkv_weight_dim_0), ParamDim(768, 768, &encoder_layers_0_self_attn_qkv_weight_dim_1)};
     params[8].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_0_self_attn_qkv_bias_dim_0)};
     params[9].shape_ptrs = {ParamDim(768, 768, &encoder_layers_0_self_attn_proj_weight_dim_0), ParamDim(768, 768, &encoder_layers_0_self_attn_proj_weight_dim_1)};
     params[10].shape_ptrs = {ParamDim(768, 768, &encoder_layers_0_self_attn_proj_bias_dim_0)};
     params[11].shape_ptrs = {ParamDim(768, 768, &encoder_layers_0_layer_norm2_weight_dim_0)};
     params[12].shape_ptrs = {ParamDim(768, 768, &encoder_layers_0_layer_norm2_bias_dim_0)};
     params[13].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_0_mlp_fc1_weight_dim_0), ParamDim(768, 768, &encoder_layers_0_mlp_fc1_weight_dim_1)};
     params[14].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_0_mlp_fc1_bias_dim_0)};
     params[15].shape_ptrs = {ParamDim(768, 768, &encoder_layers_0_mlp_fc2_weight_dim_0), ParamDim(3072, 3072, &encoder_layers_0_mlp_fc2_weight_dim_1)};
     params[16].shape_ptrs = {ParamDim(768, 768, &encoder_layers_0_mlp_fc2_bias_dim_0)};
     params[17].shape_ptrs = {ParamDim(768, 768, &encoder_layers_1_layer_norm1_weight_dim_0)};
     params[18].shape_ptrs = {ParamDim(768, 768, &encoder_layers_1_layer_norm1_bias_dim_0)};
     params[19].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_1_self_attn_qkv_weight_dim_0), ParamDim(768, 768, &encoder_layers_1_self_attn_qkv_weight_dim_1)};
     params[20].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_1_self_attn_qkv_bias_dim_0)};
     params[21].shape_ptrs = {ParamDim(768, 768, &encoder_layers_1_self_attn_proj_weight_dim_0), ParamDim(768, 768, &encoder_layers_1_self_attn_proj_weight_dim_1)};
     params[22].shape_ptrs = {ParamDim(768, 768, &encoder_layers_1_self_attn_proj_bias_dim_0)};
     params[23].shape_ptrs = {ParamDim(768, 768, &encoder_layers_1_layer_norm2_weight_dim_0)};
     params[24].shape_ptrs = {ParamDim(768, 768, &encoder_layers_1_layer_norm2_bias_dim_0)};
     params[25].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_1_mlp_fc1_weight_dim_0), ParamDim(768, 768, &encoder_layers_1_mlp_fc1_weight_dim_1)};
     params[26].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_1_mlp_fc1_bias_dim_0)};
     params[27].shape_ptrs = {ParamDim(768, 768, &encoder_layers_1_mlp_fc2_weight_dim_0), ParamDim(3072, 3072, &encoder_layers_1_mlp_fc2_weight_dim_1)};
     params[28].shape_ptrs = {ParamDim(768, 768, &encoder_layers_1_mlp_fc2_bias_dim_0)};
     params[29].shape_ptrs = {ParamDim(768, 768, &encoder_layers_2_layer_norm1_weight_dim_0)};
     params[30].shape_ptrs = {ParamDim(768, 768, &encoder_layers_2_layer_norm1_bias_dim_0)};
     params[31].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_2_self_attn_qkv_weight_dim_0), ParamDim(768, 768, &encoder_layers_2_self_attn_qkv_weight_dim_1)};
     params[32].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_2_self_attn_qkv_bias_dim_0)};
     params[33].shape_ptrs = {ParamDim(768, 768, &encoder_layers_2_self_attn_proj_weight_dim_0), ParamDim(768, 768, &encoder_layers_2_self_attn_proj_weight_dim_1)};
     params[34].shape_ptrs = {ParamDim(768, 768, &encoder_layers_2_self_attn_proj_bias_dim_0)};
     params[35].shape_ptrs = {ParamDim(768, 768, &encoder_layers_2_layer_norm2_weight_dim_0)};
     params[36].shape_ptrs = {ParamDim(768, 768, &encoder_layers_2_layer_norm2_bias_dim_0)};
     params[37].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_2_mlp_fc1_weight_dim_0), ParamDim(768, 768, &encoder_layers_2_mlp_fc1_weight_dim_1)};
     params[38].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_2_mlp_fc1_bias_dim_0)};
     params[39].shape_ptrs = {ParamDim(768, 768, &encoder_layers_2_mlp_fc2_weight_dim_0), ParamDim(3072, 3072, &encoder_layers_2_mlp_fc2_weight_dim_1)};
     params[40].shape_ptrs = {ParamDim(768, 768, &encoder_layers_2_mlp_fc2_bias_dim_0)};
     params[41].shape_ptrs = {ParamDim(768, 768, &encoder_layers_3_layer_norm1_weight_dim_0)};
     params[42].shape_ptrs = {ParamDim(768, 768, &encoder_layers_3_layer_norm1_bias_dim_0)};
     params[43].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_3_self_attn_qkv_weight_dim_0), ParamDim(768, 768, &encoder_layers_3_self_attn_qkv_weight_dim_1)};
     params[44].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_3_self_attn_qkv_bias_dim_0)};
     params[45].shape_ptrs = {ParamDim(768, 768, &encoder_layers_3_self_attn_proj_weight_dim_0), ParamDim(768, 768, &encoder_layers_3_self_attn_proj_weight_dim_1)};
     params[46].shape_ptrs = {ParamDim(768, 768, &encoder_layers_3_self_attn_proj_bias_dim_0)};
     params[47].shape_ptrs = {ParamDim(768, 768, &encoder_layers_3_layer_norm2_weight_dim_0)};
     params[48].shape_ptrs = {ParamDim(768, 768, &encoder_layers_3_layer_norm2_bias_dim_0)};
     params[49].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_3_mlp_fc1_weight_dim_0), ParamDim(768, 768, &encoder_layers_3_mlp_fc1_weight_dim_1)};
     params[50].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_3_mlp_fc1_bias_dim_0)};
     params[51].shape_ptrs = {ParamDim(768, 768, &encoder_layers_3_mlp_fc2_weight_dim_0), ParamDim(3072, 3072, &encoder_layers_3_mlp_fc2_weight_dim_1)};
     params[52].shape_ptrs = {ParamDim(768, 768, &encoder_layers_3_mlp_fc2_bias_dim_0)};
     params[53].shape_ptrs = {ParamDim(768, 768, &encoder_layers_4_layer_norm1_weight_dim_0)};
     params[54].shape_ptrs = {ParamDim(768, 768, &encoder_layers_4_layer_norm1_bias_dim_0)};
     params[55].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_4_self_attn_qkv_weight_dim_0), ParamDim(768, 768, &encoder_layers_4_self_attn_qkv_weight_dim_1)};
     params[56].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_4_self_attn_qkv_bias_dim_0)};
     params[57].shape_ptrs = {ParamDim(768, 768, &encoder_layers_4_self_attn_proj_weight_dim_0), ParamDim(768, 768, &encoder_layers_4_self_attn_proj_weight_dim_1)};
     params[58].shape_ptrs = {ParamDim(768, 768, &encoder_layers_4_self_attn_proj_bias_dim_0)};
     params[59].shape_ptrs = {ParamDim(768, 768, &encoder_layers_4_layer_norm2_weight_dim_0)};
     params[60].shape_ptrs = {ParamDim(768, 768, &encoder_layers_4_layer_norm2_bias_dim_0)};
     params[61].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_4_mlp_fc1_weight_dim_0), ParamDim(768, 768, &encoder_layers_4_mlp_fc1_weight_dim_1)};
     params[62].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_4_mlp_fc1_bias_dim_0)};
     params[63].shape_ptrs = {ParamDim(768, 768, &encoder_layers_4_mlp_fc2_weight_dim_0), ParamDim(3072, 3072, &encoder_layers_4_mlp_fc2_weight_dim_1)};
     params[64].shape_ptrs = {ParamDim(768, 768, &encoder_layers_4_mlp_fc2_bias_dim_0)};
     params[65].shape_ptrs = {ParamDim(768, 768, &encoder_layers_5_layer_norm1_weight_dim_0)};
     params[66].shape_ptrs = {ParamDim(768, 768, &encoder_layers_5_layer_norm1_bias_dim_0)};
     params[67].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_5_self_attn_qkv_weight_dim_0), ParamDim(768, 768, &encoder_layers_5_self_attn_qkv_weight_dim_1)};
     params[68].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_5_self_attn_qkv_bias_dim_0)};
     params[69].shape_ptrs = {ParamDim(768, 768, &encoder_layers_5_self_attn_proj_weight_dim_0), ParamDim(768, 768, &encoder_layers_5_self_attn_proj_weight_dim_1)};
     params[70].shape_ptrs = {ParamDim(768, 768, &encoder_layers_5_self_attn_proj_bias_dim_0)};
     params[71].shape_ptrs = {ParamDim(768, 768, &encoder_layers_5_layer_norm2_weight_dim_0)};
     params[72].shape_ptrs = {ParamDim(768, 768, &encoder_layers_5_layer_norm2_bias_dim_0)};
     params[73].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_5_mlp_fc1_weight_dim_0), ParamDim(768, 768, &encoder_layers_5_mlp_fc1_weight_dim_1)};
     params[74].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_5_mlp_fc1_bias_dim_0)};
     params[75].shape_ptrs = {ParamDim(768, 768, &encoder_layers_5_mlp_fc2_weight_dim_0), ParamDim(3072, 3072, &encoder_layers_5_mlp_fc2_weight_dim_1)};
     params[76].shape_ptrs = {ParamDim(768, 768, &encoder_layers_5_mlp_fc2_bias_dim_0)};
     params[77].shape_ptrs = {ParamDim(768, 768, &encoder_layers_6_layer_norm1_weight_dim_0)};
     params[78].shape_ptrs = {ParamDim(768, 768, &encoder_layers_6_layer_norm1_bias_dim_0)};
     params[79].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_6_self_attn_qkv_weight_dim_0), ParamDim(768, 768, &encoder_layers_6_self_attn_qkv_weight_dim_1)};
     params[80].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_6_self_attn_qkv_bias_dim_0)};
     params[81].shape_ptrs = {ParamDim(768, 768, &encoder_layers_6_self_attn_proj_weight_dim_0), ParamDim(768, 768, &encoder_layers_6_self_attn_proj_weight_dim_1)};
     params[82].shape_ptrs = {ParamDim(768, 768, &encoder_layers_6_self_attn_proj_bias_dim_0)};
     params[83].shape_ptrs = {ParamDim(768, 768, &encoder_layers_6_layer_norm2_weight_dim_0)};
     params[84].shape_ptrs = {ParamDim(768, 768, &encoder_layers_6_layer_norm2_bias_dim_0)};
     params[85].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_6_mlp_fc1_weight_dim_0), ParamDim(768, 768, &encoder_layers_6_mlp_fc1_weight_dim_1)};
     params[86].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_6_mlp_fc1_bias_dim_0)};
     params[87].shape_ptrs = {ParamDim(768, 768, &encoder_layers_6_mlp_fc2_weight_dim_0), ParamDim(3072, 3072, &encoder_layers_6_mlp_fc2_weight_dim_1)};
     params[88].shape_ptrs = {ParamDim(768, 768, &encoder_layers_6_mlp_fc2_bias_dim_0)};
     params[89].shape_ptrs = {ParamDim(768, 768, &encoder_layers_7_layer_norm1_weight_dim_0)};
     params[90].shape_ptrs = {ParamDim(768, 768, &encoder_layers_7_layer_norm1_bias_dim_0)};
     params[91].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_7_self_attn_qkv_weight_dim_0), ParamDim(768, 768, &encoder_layers_7_self_attn_qkv_weight_dim_1)};
     params[92].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_7_self_attn_qkv_bias_dim_0)};
     params[93].shape_ptrs = {ParamDim(768, 768, &encoder_layers_7_self_attn_proj_weight_dim_0), ParamDim(768, 768, &encoder_layers_7_self_attn_proj_weight_dim_1)};
     params[94].shape_ptrs = {ParamDim(768, 768, &encoder_layers_7_self_attn_proj_bias_dim_0)};
     params[95].shape_ptrs = {ParamDim(768, 768, &encoder_layers_7_layer_norm2_weight_dim_0)};
     params[96].shape_ptrs = {ParamDim(768, 768, &encoder_layers_7_layer_norm2_bias_dim_0)};
     params[97].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_7_mlp_fc1_weight_dim_0), ParamDim(768, 768, &encoder_layers_7_mlp_fc1_weight_dim_1)};
     params[98].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_7_mlp_fc1_bias_dim_0)};
     params[99].shape_ptrs = {ParamDim(768, 768, &encoder_layers_7_mlp_fc2_weight_dim_0), ParamDim(3072, 3072, &encoder_layers_7_mlp_fc2_weight_dim_1)};
     params[100].shape_ptrs = {ParamDim(768, 768, &encoder_layers_7_mlp_fc2_bias_dim_0)};
     params[101].shape_ptrs = {ParamDim(768, 768, &encoder_layers_8_layer_norm1_weight_dim_0)};
     params[102].shape_ptrs = {ParamDim(768, 768, &encoder_layers_8_layer_norm1_bias_dim_0)};
     params[103].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_8_self_attn_qkv_weight_dim_0), ParamDim(768, 768, &encoder_layers_8_self_attn_qkv_weight_dim_1)};
     params[104].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_8_self_attn_qkv_bias_dim_0)};
     params[105].shape_ptrs = {ParamDim(768, 768, &encoder_layers_8_self_attn_proj_weight_dim_0), ParamDim(768, 768, &encoder_layers_8_self_attn_proj_weight_dim_1)};
     params[106].shape_ptrs = {ParamDim(768, 768, &encoder_layers_8_self_attn_proj_bias_dim_0)};
     params[107].shape_ptrs = {ParamDim(768, 768, &encoder_layers_8_layer_norm2_weight_dim_0)};
     params[108].shape_ptrs = {ParamDim(768, 768, &encoder_layers_8_layer_norm2_bias_dim_0)};
     params[109].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_8_mlp_fc1_weight_dim_0), ParamDim(768, 768, &encoder_layers_8_mlp_fc1_weight_dim_1)};
     params[110].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_8_mlp_fc1_bias_dim_0)};
     params[111].shape_ptrs = {ParamDim(768, 768, &encoder_layers_8_mlp_fc2_weight_dim_0), ParamDim(3072, 3072, &encoder_layers_8_mlp_fc2_weight_dim_1)};
     params[112].shape_ptrs = {ParamDim(768, 768, &encoder_layers_8_mlp_fc2_bias_dim_0)};
     params[113].shape_ptrs = {ParamDim(768, 768, &encoder_layers_9_layer_norm1_weight_dim_0)};
     params[114].shape_ptrs = {ParamDim(768, 768, &encoder_layers_9_layer_norm1_bias_dim_0)};
     params[115].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_9_self_attn_qkv_weight_dim_0), ParamDim(768, 768, &encoder_layers_9_self_attn_qkv_weight_dim_1)};
     params[116].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_9_self_attn_qkv_bias_dim_0)};
     params[117].shape_ptrs = {ParamDim(768, 768, &encoder_layers_9_self_attn_proj_weight_dim_0), ParamDim(768, 768, &encoder_layers_9_self_attn_proj_weight_dim_1)};
     params[118].shape_ptrs = {ParamDim(768, 768, &encoder_layers_9_self_attn_proj_bias_dim_0)};
     params[119].shape_ptrs = {ParamDim(768, 768, &encoder_layers_9_layer_norm2_weight_dim_0)};
     params[120].shape_ptrs = {ParamDim(768, 768, &encoder_layers_9_layer_norm2_bias_dim_0)};
     params[121].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_9_mlp_fc1_weight_dim_0), ParamDim(768, 768, &encoder_layers_9_mlp_fc1_weight_dim_1)};
     params[122].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_9_mlp_fc1_bias_dim_0)};
     params[123].shape_ptrs = {ParamDim(768, 768, &encoder_layers_9_mlp_fc2_weight_dim_0), ParamDim(3072, 3072, &encoder_layers_9_mlp_fc2_weight_dim_1)};
     params[124].shape_ptrs = {ParamDim(768, 768, &encoder_layers_9_mlp_fc2_bias_dim_0)};
     params[125].shape_ptrs = {ParamDim(768, 768, &encoder_layers_10_layer_norm1_weight_dim_0)};
     params[126].shape_ptrs = {ParamDim(768, 768, &encoder_layers_10_layer_norm1_bias_dim_0)};
     params[127].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_10_self_attn_qkv_weight_dim_0), ParamDim(768, 768, &encoder_layers_10_self_attn_qkv_weight_dim_1)};
     params[128].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_10_self_attn_qkv_bias_dim_0)};
     params[129].shape_ptrs = {ParamDim(768, 768, &encoder_layers_10_self_attn_proj_weight_dim_0), ParamDim(768, 768, &encoder_layers_10_self_attn_proj_weight_dim_1)};
     params[130].shape_ptrs = {ParamDim(768, 768, &encoder_layers_10_self_attn_proj_bias_dim_0)};
     params[131].shape_ptrs = {ParamDim(768, 768, &encoder_layers_10_layer_norm2_weight_dim_0)};
     params[132].shape_ptrs = {ParamDim(768, 768, &encoder_layers_10_layer_norm2_bias_dim_0)};
     params[133].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_10_mlp_fc1_weight_dim_0), ParamDim(768, 768, &encoder_layers_10_mlp_fc1_weight_dim_1)};
     params[134].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_10_mlp_fc1_bias_dim_0)};
     params[135].shape_ptrs = {ParamDim(768, 768, &encoder_layers_10_mlp_fc2_weight_dim_0), ParamDim(3072, 3072, &encoder_layers_10_mlp_fc2_weight_dim_1)};
     params[136].shape_ptrs = {ParamDim(768, 768, &encoder_layers_10_mlp_fc2_bias_dim_0)};
     params[137].shape_ptrs = {ParamDim(768, 768, &encoder_layers_11_layer_norm1_weight_dim_0)};
     params[138].shape_ptrs = {ParamDim(768, 768, &encoder_layers_11_layer_norm1_bias_dim_0)};
     params[139].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_11_self_attn_qkv_weight_dim_0), ParamDim(768, 768, &encoder_layers_11_self_attn_qkv_weight_dim_1)};
     params[140].shape_ptrs = {ParamDim(2304, 2304, &encoder_layers_11_self_attn_qkv_bias_dim_0)};
     params[141].shape_ptrs = {ParamDim(768, 768, &encoder_layers_11_self_attn_proj_weight_dim_0), ParamDim(768, 768, &encoder_layers_11_self_attn_proj_weight_dim_1)};
     params[142].shape_ptrs = {ParamDim(768, 768, &encoder_layers_11_self_attn_proj_bias_dim_0)};
     params[143].shape_ptrs = {ParamDim(768, 768, &encoder_layers_11_layer_norm2_weight_dim_0)};
     params[144].shape_ptrs = {ParamDim(768, 768, &encoder_layers_11_layer_norm2_bias_dim_0)};
     params[145].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_11_mlp_fc1_weight_dim_0), ParamDim(768, 768, &encoder_layers_11_mlp_fc1_weight_dim_1)};
     params[146].shape_ptrs = {ParamDim(3072, 3072, &encoder_layers_11_mlp_fc1_bias_dim_0)};
     params[147].shape_ptrs = {ParamDim(768, 768, &encoder_layers_11_mlp_fc2_weight_dim_0), ParamDim(3072, 3072, &encoder_layers_11_mlp_fc2_weight_dim_1)};
     params[148].shape_ptrs = {ParamDim(768, 768, &encoder_layers_11_mlp_fc2_bias_dim_0)};
     params[149].shape_ptrs = {ParamDim(768, 768, &final_layer_norm_weight_dim_0)};
     params[150].shape_ptrs = {ParamDim(768, 768, &final_layer_norm_bias_dim_0)};
     params[2].shape_ptrs = {ParamDim(1, 1, &reshape_174_0_dim_0), ParamDim(64, 64, &reshape_174_0_dim_1), ParamDim(768, 768, &reshape_174_0_dim_2)};
    }

    ~Model() {
      DestroyEvent(run_finished);
      StreamDestroy(graph_capture_stream);
      if (graph_exec != nullptr) {
        GraphExecDestroy(graph_exec);
      }
      if (graph != nullptr) {
        GraphDestroy(graph);
      }
    }

    Model(Model&&) = default;
    Model& operator=(Model&&) = default;

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void SetUpInputsOutputs() {
             input0 = static_cast<decltype(input0)>(params[0].ptr);

if (input0 == nullptr) {
    throw std::runtime_error("Constant input0 was not set! Set the value with set_constant.");
}
    
     reshape_0_0 = input0;

if (embeddings_token_embedding_weight == nullptr) {
    throw std::runtime_error("Constant embeddings_token_embedding_weight was not set! Set the value with set_constant.");
}
    
     input1 = static_cast<decltype(input1)>(params[1].ptr);

if (input1 == nullptr) {
    throw std::runtime_error("Constant input1 was not set! Set the value with set_constant.");
}
    
     reshape_2_0 = input1;

if (embeddings_position_embedding_weight == nullptr) {
    throw std::runtime_error("Constant embeddings_position_embedding_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_0_layer_norm1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_0_layer_norm1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_0_layer_norm1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_0_layer_norm1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_0_self_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_0_self_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_0_self_attn_qkv_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_0_self_attn_qkv_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_0_self_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_0_self_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_0_self_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_0_self_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_0_layer_norm2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_0_layer_norm2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_0_layer_norm2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_0_layer_norm2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_0_mlp_fc1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_0_mlp_fc1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_0_mlp_fc1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_0_mlp_fc1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_0_mlp_fc2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_0_mlp_fc2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_0_mlp_fc2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_0_mlp_fc2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_1_layer_norm1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_1_layer_norm1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_1_layer_norm1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_1_layer_norm1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_1_self_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_1_self_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_1_self_attn_qkv_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_1_self_attn_qkv_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_1_self_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_1_self_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_1_self_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_1_self_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_1_layer_norm2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_1_layer_norm2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_1_layer_norm2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_1_layer_norm2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_1_mlp_fc1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_1_mlp_fc1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_1_mlp_fc1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_1_mlp_fc1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_1_mlp_fc2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_1_mlp_fc2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_1_mlp_fc2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_1_mlp_fc2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_2_layer_norm1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_2_layer_norm1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_2_layer_norm1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_2_layer_norm1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_2_self_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_2_self_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_2_self_attn_qkv_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_2_self_attn_qkv_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_2_self_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_2_self_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_2_self_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_2_self_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_2_layer_norm2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_2_layer_norm2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_2_layer_norm2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_2_layer_norm2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_2_mlp_fc1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_2_mlp_fc1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_2_mlp_fc1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_2_mlp_fc1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_2_mlp_fc2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_2_mlp_fc2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_2_mlp_fc2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_2_mlp_fc2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_3_layer_norm1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_3_layer_norm1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_3_layer_norm1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_3_layer_norm1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_3_self_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_3_self_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_3_self_attn_qkv_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_3_self_attn_qkv_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_3_self_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_3_self_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_3_self_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_3_self_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_3_layer_norm2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_3_layer_norm2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_3_layer_norm2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_3_layer_norm2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_3_mlp_fc1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_3_mlp_fc1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_3_mlp_fc1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_3_mlp_fc1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_3_mlp_fc2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_3_mlp_fc2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_3_mlp_fc2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_3_mlp_fc2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_4_layer_norm1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_4_layer_norm1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_4_layer_norm1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_4_layer_norm1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_4_self_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_4_self_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_4_self_attn_qkv_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_4_self_attn_qkv_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_4_self_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_4_self_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_4_self_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_4_self_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_4_layer_norm2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_4_layer_norm2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_4_layer_norm2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_4_layer_norm2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_4_mlp_fc1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_4_mlp_fc1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_4_mlp_fc1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_4_mlp_fc1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_4_mlp_fc2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_4_mlp_fc2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_4_mlp_fc2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_4_mlp_fc2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_5_layer_norm1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_5_layer_norm1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_5_layer_norm1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_5_layer_norm1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_5_self_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_5_self_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_5_self_attn_qkv_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_5_self_attn_qkv_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_5_self_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_5_self_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_5_self_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_5_self_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_5_layer_norm2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_5_layer_norm2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_5_layer_norm2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_5_layer_norm2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_5_mlp_fc1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_5_mlp_fc1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_5_mlp_fc1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_5_mlp_fc1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_5_mlp_fc2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_5_mlp_fc2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_5_mlp_fc2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_5_mlp_fc2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_6_layer_norm1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_6_layer_norm1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_6_layer_norm1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_6_layer_norm1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_6_self_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_6_self_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_6_self_attn_qkv_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_6_self_attn_qkv_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_6_self_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_6_self_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_6_self_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_6_self_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_6_layer_norm2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_6_layer_norm2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_6_layer_norm2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_6_layer_norm2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_6_mlp_fc1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_6_mlp_fc1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_6_mlp_fc1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_6_mlp_fc1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_6_mlp_fc2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_6_mlp_fc2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_6_mlp_fc2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_6_mlp_fc2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_7_layer_norm1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_7_layer_norm1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_7_layer_norm1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_7_layer_norm1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_7_self_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_7_self_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_7_self_attn_qkv_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_7_self_attn_qkv_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_7_self_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_7_self_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_7_self_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_7_self_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_7_layer_norm2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_7_layer_norm2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_7_layer_norm2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_7_layer_norm2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_7_mlp_fc1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_7_mlp_fc1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_7_mlp_fc1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_7_mlp_fc1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_7_mlp_fc2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_7_mlp_fc2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_7_mlp_fc2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_7_mlp_fc2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_8_layer_norm1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_8_layer_norm1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_8_layer_norm1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_8_layer_norm1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_8_self_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_8_self_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_8_self_attn_qkv_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_8_self_attn_qkv_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_8_self_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_8_self_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_8_self_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_8_self_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_8_layer_norm2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_8_layer_norm2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_8_layer_norm2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_8_layer_norm2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_8_mlp_fc1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_8_mlp_fc1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_8_mlp_fc1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_8_mlp_fc1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_8_mlp_fc2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_8_mlp_fc2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_8_mlp_fc2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_8_mlp_fc2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_9_layer_norm1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_9_layer_norm1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_9_layer_norm1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_9_layer_norm1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_9_self_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_9_self_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_9_self_attn_qkv_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_9_self_attn_qkv_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_9_self_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_9_self_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_9_self_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_9_self_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_9_layer_norm2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_9_layer_norm2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_9_layer_norm2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_9_layer_norm2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_9_mlp_fc1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_9_mlp_fc1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_9_mlp_fc1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_9_mlp_fc1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_9_mlp_fc2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_9_mlp_fc2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_9_mlp_fc2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_9_mlp_fc2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_10_layer_norm1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_10_layer_norm1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_10_layer_norm1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_10_layer_norm1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_10_self_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_10_self_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_10_self_attn_qkv_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_10_self_attn_qkv_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_10_self_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_10_self_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_10_self_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_10_self_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_10_layer_norm2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_10_layer_norm2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_10_layer_norm2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_10_layer_norm2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_10_mlp_fc1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_10_mlp_fc1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_10_mlp_fc1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_10_mlp_fc1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_10_mlp_fc2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_10_mlp_fc2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_10_mlp_fc2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_10_mlp_fc2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_11_layer_norm1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_11_layer_norm1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_11_layer_norm1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_11_layer_norm1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_11_self_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_11_self_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_11_self_attn_qkv_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_11_self_attn_qkv_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_11_self_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_11_self_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_11_self_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_11_self_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_11_layer_norm2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_11_layer_norm2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_11_layer_norm2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_11_layer_norm2_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_11_mlp_fc1_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_11_mlp_fc1_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_11_mlp_fc1_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_11_mlp_fc1_bias was not set! Set the value with set_constant.");
}
    

if (encoder_layers_11_mlp_fc2_weight == nullptr) {
    throw std::runtime_error("Constant encoder_layers_11_mlp_fc2_weight was not set! Set the value with set_constant.");
}
    

if (encoder_layers_11_mlp_fc2_bias == nullptr) {
    throw std::runtime_error("Constant encoder_layers_11_mlp_fc2_bias was not set! Set the value with set_constant.");
}
    

if (final_layer_norm_weight == nullptr) {
    throw std::runtime_error("Constant final_layer_norm_weight was not set! Set the value with set_constant.");
}
    

if (final_layer_norm_bias == nullptr) {
    throw std::runtime_error("Constant final_layer_norm_bias was not set! Set the value with set_constant.");
}
    
     output_0 = static_cast<decltype(output_0)>(params[2].ptr);

if (output_0 == nullptr) {
    throw std::runtime_error("Constant output_0 was not set! Set the value with set_constant.");
}
    
    }

    void DeviceToDeviceCopies(StreamType stream) {
  
    }

    void Run(StreamType stream, bool graph_mode) {
      SetUpInputsOutputs();
      if (target_has_graph_mode && graph_mode) {
        RunAsGraph(stream);
      } else {
        RunImpl(stream);
      }
      DEVICE_CHECK(EventRecord(run_finished, stream));
    }

    void RunImpl(StreamType stream) {
  
  
    reshape_0(
        &input0_dim_0,
        &input0_dim_1,
        &reshape_0_0_dim_0
    );
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    batch_gather_1(
       reinterpret_cast<half*>(
        &(batch_gather_1_0->raw())), reinterpret_cast<half*>(
        &(embeddings_token_embedding_weight->raw())), reinterpret_cast<int64_t*>(reshape_0_0),
        1,
        64,
        768,
        49408,
        global_workspace, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    reshape_0(
        &input1_dim_0,
        &input1_dim_1,
        &reshape_2_0_dim_0
    );
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    batch_gather_3(
       reinterpret_cast<half*>(
        &(batch_gather_3_0->raw())), reinterpret_cast<half*>(
        &(embeddings_position_embedding_weight->raw())), reinterpret_cast<int64_t*>(reshape_2_0),
        1,
        64,
        768,
        77,
        global_workspace, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_176_n_elements = 64 * 768;
        invoke_fused_elementwise_176(reinterpret_cast<half*>(reshape_6_0), reinterpret_cast<half*>(batch_gather_3_0),reinterpret_cast<half*>(batch_gather_1_0),  fused_elementwise_176_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_6_0_dim_0;

        M *= reshape_6_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_6_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_7_0->raw())), reinterpret_cast<half*>(&(reshape_6_0->raw())), reinterpret_cast<half*>(&(encoder_layers_0_layer_norm1_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_0_layer_norm1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_8(
        layernorm_7_0,
        encoder_layers_0_self_attn_qkv_weight,

        encoder_layers_0_self_attn_qkv_bias,

        reshape_9_0,
        global_workspace,
        1,

        &reshape_6_0_dim_0,

        &reshape_6_0_dim_1,

        &reshape_6_0_dim_2,


        &encoder_layers_0_self_attn_qkv_weight_dim_0,

        &encoder_layers_0_self_attn_qkv_weight_dim_1,


        &reshape_6_0_dim_0,

        &reshape_6_0_dim_1,

        &encoder_layers_0_self_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_10_0->raw())), reinterpret_cast<half*>(&(reshape_9_0->raw())), reinterpret_cast<int*>(encoder_layers_0_self_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_10_0,
        encoder_layers_0_self_attn_proj_weight,
        encoder_layers_0_self_attn_proj_bias,
        reshape_6_0,

        reshape_13_0,
        global_workspace,

     1,


        &reshape_11_0_dim_0,

        &reshape_11_0_dim_1,


        &encoder_layers_0_self_attn_proj_weight_dim_0,

        &encoder_layers_0_self_attn_proj_weight_dim_1,


        &reshape_11_0_dim_0,

        &encoder_layers_0_self_attn_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_13_0_dim_0;

        M *= reshape_13_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_13_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_14_0->raw())), reinterpret_cast<half*>(&(reshape_13_0->raw())), reinterpret_cast<half*>(&(encoder_layers_0_layer_norm2_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_0_layer_norm2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_14_0,
        encoder_layers_0_mlp_fc1_weight,

        encoder_layers_0_mlp_fc1_bias,

        gemm_rcr_bias_15_0,
        global_workspace,
        1,

        &reshape_13_0_dim_0,

        &reshape_13_0_dim_1,

        &reshape_13_0_dim_2,


        &encoder_layers_0_mlp_fc1_weight_dim_0,

        &encoder_layers_0_mlp_fc1_weight_dim_1,


        &reshape_13_0_dim_0,

        &reshape_13_0_dim_1,

        &encoder_layers_0_mlp_fc1_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_177_n_elements = 1 * 64 * 3072;
        invoke_fused_elementwise_177(reinterpret_cast<half*>(elementwise_18_0), reinterpret_cast<half*>(gemm_rcr_bias_15_0),  fused_elementwise_177_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_19(
        elementwise_18_0,
        encoder_layers_0_mlp_fc2_weight,
        encoder_layers_0_mlp_fc2_bias,
        reshape_13_0,

        reshape_20_0,
        global_workspace,

     2,


        &reshape_13_0_dim_0,

        &reshape_13_0_dim_1,

        &encoder_layers_0_mlp_fc1_weight_dim_0,


        &encoder_layers_0_mlp_fc2_weight_dim_0,

        &encoder_layers_0_mlp_fc2_weight_dim_1,


        &reshape_13_0_dim_0,

        &reshape_13_0_dim_1,

        &encoder_layers_0_mlp_fc2_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_20_0_dim_0;

        M *= reshape_20_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_20_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_21_0->raw())), reinterpret_cast<half*>(&(reshape_20_0->raw())), reinterpret_cast<half*>(&(encoder_layers_1_layer_norm1_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_1_layer_norm1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_8(
        layernorm_21_0,
        encoder_layers_1_self_attn_qkv_weight,

        encoder_layers_1_self_attn_qkv_bias,

        reshape_23_0,
        global_workspace,
        1,

        &reshape_20_0_dim_0,

        &reshape_20_0_dim_1,

        &reshape_20_0_dim_2,


        &encoder_layers_1_self_attn_qkv_weight_dim_0,

        &encoder_layers_1_self_attn_qkv_weight_dim_1,


        &reshape_20_0_dim_0,

        &reshape_20_0_dim_1,

        &encoder_layers_1_self_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_24_0->raw())), reinterpret_cast<half*>(&(reshape_23_0->raw())), reinterpret_cast<int*>(encoder_layers_1_self_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_24_0,
        encoder_layers_1_self_attn_proj_weight,
        encoder_layers_1_self_attn_proj_bias,
        reshape_20_0,

        reshape_27_0,
        global_workspace,

     1,


        &reshape_25_0_dim_0,

        &reshape_25_0_dim_1,


        &encoder_layers_1_self_attn_proj_weight_dim_0,

        &encoder_layers_1_self_attn_proj_weight_dim_1,


        &reshape_25_0_dim_0,

        &encoder_layers_1_self_attn_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_27_0_dim_0;

        M *= reshape_27_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_27_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_28_0->raw())), reinterpret_cast<half*>(&(reshape_27_0->raw())), reinterpret_cast<half*>(&(encoder_layers_1_layer_norm2_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_1_layer_norm2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_28_0,
        encoder_layers_1_mlp_fc1_weight,

        encoder_layers_1_mlp_fc1_bias,

        gemm_rcr_bias_29_0,
        global_workspace,
        1,

        &reshape_27_0_dim_0,

        &reshape_27_0_dim_1,

        &reshape_27_0_dim_2,


        &encoder_layers_1_mlp_fc1_weight_dim_0,

        &encoder_layers_1_mlp_fc1_weight_dim_1,


        &reshape_27_0_dim_0,

        &reshape_27_0_dim_1,

        &encoder_layers_1_mlp_fc1_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_178_n_elements = 1 * 64 * 3072;
        invoke_fused_elementwise_178(reinterpret_cast<half*>(elementwise_32_0), reinterpret_cast<half*>(gemm_rcr_bias_29_0),  fused_elementwise_178_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_19(
        elementwise_32_0,
        encoder_layers_1_mlp_fc2_weight,
        encoder_layers_1_mlp_fc2_bias,
        reshape_27_0,

        reshape_34_0,
        global_workspace,

     2,


        &reshape_27_0_dim_0,

        &reshape_27_0_dim_1,

        &encoder_layers_1_mlp_fc1_weight_dim_0,


        &encoder_layers_1_mlp_fc2_weight_dim_0,

        &encoder_layers_1_mlp_fc2_weight_dim_1,


        &reshape_27_0_dim_0,

        &reshape_27_0_dim_1,

        &encoder_layers_1_mlp_fc2_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_34_0_dim_0;

        M *= reshape_34_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_34_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_35_0->raw())), reinterpret_cast<half*>(&(reshape_34_0->raw())), reinterpret_cast<half*>(&(encoder_layers_2_layer_norm1_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_2_layer_norm1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_8(
        layernorm_35_0,
        encoder_layers_2_self_attn_qkv_weight,

        encoder_layers_2_self_attn_qkv_bias,

        reshape_37_0,
        global_workspace,
        1,

        &reshape_34_0_dim_0,

        &reshape_34_0_dim_1,

        &reshape_34_0_dim_2,


        &encoder_layers_2_self_attn_qkv_weight_dim_0,

        &encoder_layers_2_self_attn_qkv_weight_dim_1,


        &reshape_34_0_dim_0,

        &reshape_34_0_dim_1,

        &encoder_layers_2_self_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_38_0->raw())), reinterpret_cast<half*>(&(reshape_37_0->raw())), reinterpret_cast<int*>(encoder_layers_2_self_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_38_0,
        encoder_layers_2_self_attn_proj_weight,
        encoder_layers_2_self_attn_proj_bias,
        reshape_34_0,

        reshape_41_0,
        global_workspace,

     1,


        &reshape_39_0_dim_0,

        &reshape_39_0_dim_1,


        &encoder_layers_2_self_attn_proj_weight_dim_0,

        &encoder_layers_2_self_attn_proj_weight_dim_1,


        &reshape_39_0_dim_0,

        &encoder_layers_2_self_attn_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_41_0_dim_0;

        M *= reshape_41_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_41_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_42_0->raw())), reinterpret_cast<half*>(&(reshape_41_0->raw())), reinterpret_cast<half*>(&(encoder_layers_2_layer_norm2_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_2_layer_norm2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_42_0,
        encoder_layers_2_mlp_fc1_weight,

        encoder_layers_2_mlp_fc1_bias,

        gemm_rcr_bias_43_0,
        global_workspace,
        1,

        &reshape_41_0_dim_0,

        &reshape_41_0_dim_1,

        &reshape_41_0_dim_2,


        &encoder_layers_2_mlp_fc1_weight_dim_0,

        &encoder_layers_2_mlp_fc1_weight_dim_1,


        &reshape_41_0_dim_0,

        &reshape_41_0_dim_1,

        &encoder_layers_2_mlp_fc1_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_179_n_elements = 1 * 64 * 3072;
        invoke_fused_elementwise_179(reinterpret_cast<half*>(elementwise_46_0), reinterpret_cast<half*>(gemm_rcr_bias_43_0),  fused_elementwise_179_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_19(
        elementwise_46_0,
        encoder_layers_2_mlp_fc2_weight,
        encoder_layers_2_mlp_fc2_bias,
        reshape_41_0,

        reshape_48_0,
        global_workspace,

     2,


        &reshape_41_0_dim_0,

        &reshape_41_0_dim_1,

        &encoder_layers_2_mlp_fc1_weight_dim_0,


        &encoder_layers_2_mlp_fc2_weight_dim_0,

        &encoder_layers_2_mlp_fc2_weight_dim_1,


        &reshape_41_0_dim_0,

        &reshape_41_0_dim_1,

        &encoder_layers_2_mlp_fc2_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_48_0_dim_0;

        M *= reshape_48_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_48_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_49_0->raw())), reinterpret_cast<half*>(&(reshape_48_0->raw())), reinterpret_cast<half*>(&(encoder_layers_3_layer_norm1_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_3_layer_norm1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_8(
        layernorm_49_0,
        encoder_layers_3_self_attn_qkv_weight,

        encoder_layers_3_self_attn_qkv_bias,

        reshape_51_0,
        global_workspace,
        1,

        &reshape_48_0_dim_0,

        &reshape_48_0_dim_1,

        &reshape_48_0_dim_2,


        &encoder_layers_3_self_attn_qkv_weight_dim_0,

        &encoder_layers_3_self_attn_qkv_weight_dim_1,


        &reshape_48_0_dim_0,

        &reshape_48_0_dim_1,

        &encoder_layers_3_self_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_52_0->raw())), reinterpret_cast<half*>(&(reshape_51_0->raw())), reinterpret_cast<int*>(encoder_layers_3_self_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_52_0,
        encoder_layers_3_self_attn_proj_weight,
        encoder_layers_3_self_attn_proj_bias,
        reshape_48_0,

        reshape_55_0,
        global_workspace,

     1,


        &reshape_53_0_dim_0,

        &reshape_53_0_dim_1,


        &encoder_layers_3_self_attn_proj_weight_dim_0,

        &encoder_layers_3_self_attn_proj_weight_dim_1,


        &reshape_53_0_dim_0,

        &encoder_layers_3_self_attn_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_55_0_dim_0;

        M *= reshape_55_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_55_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_56_0->raw())), reinterpret_cast<half*>(&(reshape_55_0->raw())), reinterpret_cast<half*>(&(encoder_layers_3_layer_norm2_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_3_layer_norm2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_56_0,
        encoder_layers_3_mlp_fc1_weight,

        encoder_layers_3_mlp_fc1_bias,

        gemm_rcr_bias_57_0,
        global_workspace,
        1,

        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &reshape_55_0_dim_2,


        &encoder_layers_3_mlp_fc1_weight_dim_0,

        &encoder_layers_3_mlp_fc1_weight_dim_1,


        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &encoder_layers_3_mlp_fc1_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_180_n_elements = 1 * 64 * 3072;
        invoke_fused_elementwise_180(reinterpret_cast<half*>(elementwise_60_0), reinterpret_cast<half*>(gemm_rcr_bias_57_0),  fused_elementwise_180_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_19(
        elementwise_60_0,
        encoder_layers_3_mlp_fc2_weight,
        encoder_layers_3_mlp_fc2_bias,
        reshape_55_0,

        reshape_62_0,
        global_workspace,

     2,


        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &encoder_layers_3_mlp_fc1_weight_dim_0,


        &encoder_layers_3_mlp_fc2_weight_dim_0,

        &encoder_layers_3_mlp_fc2_weight_dim_1,


        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &encoder_layers_3_mlp_fc2_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_62_0_dim_0;

        M *= reshape_62_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_62_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_63_0->raw())), reinterpret_cast<half*>(&(reshape_62_0->raw())), reinterpret_cast<half*>(&(encoder_layers_4_layer_norm1_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_4_layer_norm1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_8(
        layernorm_63_0,
        encoder_layers_4_self_attn_qkv_weight,

        encoder_layers_4_self_attn_qkv_bias,

        reshape_65_0,
        global_workspace,
        1,

        &reshape_62_0_dim_0,

        &reshape_62_0_dim_1,

        &reshape_62_0_dim_2,


        &encoder_layers_4_self_attn_qkv_weight_dim_0,

        &encoder_layers_4_self_attn_qkv_weight_dim_1,


        &reshape_62_0_dim_0,

        &reshape_62_0_dim_1,

        &encoder_layers_4_self_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_66_0->raw())), reinterpret_cast<half*>(&(reshape_65_0->raw())), reinterpret_cast<int*>(encoder_layers_4_self_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_66_0,
        encoder_layers_4_self_attn_proj_weight,
        encoder_layers_4_self_attn_proj_bias,
        reshape_62_0,

        reshape_69_0,
        global_workspace,

     1,


        &reshape_67_0_dim_0,

        &reshape_67_0_dim_1,


        &encoder_layers_4_self_attn_proj_weight_dim_0,

        &encoder_layers_4_self_attn_proj_weight_dim_1,


        &reshape_67_0_dim_0,

        &encoder_layers_4_self_attn_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_69_0_dim_0;

        M *= reshape_69_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_69_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_70_0->raw())), reinterpret_cast<half*>(&(reshape_69_0->raw())), reinterpret_cast<half*>(&(encoder_layers_4_layer_norm2_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_4_layer_norm2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_70_0,
        encoder_layers_4_mlp_fc1_weight,

        encoder_layers_4_mlp_fc1_bias,

        gemm_rcr_bias_71_0,
        global_workspace,
        1,

        &reshape_69_0_dim_0,

        &reshape_69_0_dim_1,

        &reshape_69_0_dim_2,


        &encoder_layers_4_mlp_fc1_weight_dim_0,

        &encoder_layers_4_mlp_fc1_weight_dim_1,


        &reshape_69_0_dim_0,

        &reshape_69_0_dim_1,

        &encoder_layers_4_mlp_fc1_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_181_n_elements = 1 * 64 * 3072;
        invoke_fused_elementwise_181(reinterpret_cast<half*>(elementwise_74_0), reinterpret_cast<half*>(gemm_rcr_bias_71_0),  fused_elementwise_181_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_19(
        elementwise_74_0,
        encoder_layers_4_mlp_fc2_weight,
        encoder_layers_4_mlp_fc2_bias,
        reshape_69_0,

        reshape_76_0,
        global_workspace,

     2,


        &reshape_69_0_dim_0,

        &reshape_69_0_dim_1,

        &encoder_layers_4_mlp_fc1_weight_dim_0,


        &encoder_layers_4_mlp_fc2_weight_dim_0,

        &encoder_layers_4_mlp_fc2_weight_dim_1,


        &reshape_69_0_dim_0,

        &reshape_69_0_dim_1,

        &encoder_layers_4_mlp_fc2_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_76_0_dim_0;

        M *= reshape_76_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_76_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_77_0->raw())), reinterpret_cast<half*>(&(reshape_76_0->raw())), reinterpret_cast<half*>(&(encoder_layers_5_layer_norm1_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_5_layer_norm1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_8(
        layernorm_77_0,
        encoder_layers_5_self_attn_qkv_weight,

        encoder_layers_5_self_attn_qkv_bias,

        reshape_79_0,
        global_workspace,
        1,

        &reshape_76_0_dim_0,

        &reshape_76_0_dim_1,

        &reshape_76_0_dim_2,


        &encoder_layers_5_self_attn_qkv_weight_dim_0,

        &encoder_layers_5_self_attn_qkv_weight_dim_1,


        &reshape_76_0_dim_0,

        &reshape_76_0_dim_1,

        &encoder_layers_5_self_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_80_0->raw())), reinterpret_cast<half*>(&(reshape_79_0->raw())), reinterpret_cast<int*>(encoder_layers_5_self_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_80_0,
        encoder_layers_5_self_attn_proj_weight,
        encoder_layers_5_self_attn_proj_bias,
        reshape_76_0,

        reshape_83_0,
        global_workspace,

     1,


        &reshape_81_0_dim_0,

        &reshape_81_0_dim_1,


        &encoder_layers_5_self_attn_proj_weight_dim_0,

        &encoder_layers_5_self_attn_proj_weight_dim_1,


        &reshape_81_0_dim_0,

        &encoder_layers_5_self_attn_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_83_0_dim_0;

        M *= reshape_83_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_83_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_84_0->raw())), reinterpret_cast<half*>(&(reshape_83_0->raw())), reinterpret_cast<half*>(&(encoder_layers_5_layer_norm2_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_5_layer_norm2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_84_0,
        encoder_layers_5_mlp_fc1_weight,

        encoder_layers_5_mlp_fc1_bias,

        gemm_rcr_bias_85_0,
        global_workspace,
        1,

        &reshape_83_0_dim_0,

        &reshape_83_0_dim_1,

        &reshape_83_0_dim_2,


        &encoder_layers_5_mlp_fc1_weight_dim_0,

        &encoder_layers_5_mlp_fc1_weight_dim_1,


        &reshape_83_0_dim_0,

        &reshape_83_0_dim_1,

        &encoder_layers_5_mlp_fc1_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_182_n_elements = 1 * 64 * 3072;
        invoke_fused_elementwise_182(reinterpret_cast<half*>(elementwise_88_0), reinterpret_cast<half*>(gemm_rcr_bias_85_0),  fused_elementwise_182_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_19(
        elementwise_88_0,
        encoder_layers_5_mlp_fc2_weight,
        encoder_layers_5_mlp_fc2_bias,
        reshape_83_0,

        reshape_90_0,
        global_workspace,

     2,


        &reshape_83_0_dim_0,

        &reshape_83_0_dim_1,

        &encoder_layers_5_mlp_fc1_weight_dim_0,


        &encoder_layers_5_mlp_fc2_weight_dim_0,

        &encoder_layers_5_mlp_fc2_weight_dim_1,


        &reshape_83_0_dim_0,

        &reshape_83_0_dim_1,

        &encoder_layers_5_mlp_fc2_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_90_0_dim_0;

        M *= reshape_90_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_90_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_91_0->raw())), reinterpret_cast<half*>(&(reshape_90_0->raw())), reinterpret_cast<half*>(&(encoder_layers_6_layer_norm1_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_6_layer_norm1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_8(
        layernorm_91_0,
        encoder_layers_6_self_attn_qkv_weight,

        encoder_layers_6_self_attn_qkv_bias,

        reshape_93_0,
        global_workspace,
        1,

        &reshape_90_0_dim_0,

        &reshape_90_0_dim_1,

        &reshape_90_0_dim_2,


        &encoder_layers_6_self_attn_qkv_weight_dim_0,

        &encoder_layers_6_self_attn_qkv_weight_dim_1,


        &reshape_90_0_dim_0,

        &reshape_90_0_dim_1,

        &encoder_layers_6_self_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_94_0->raw())), reinterpret_cast<half*>(&(reshape_93_0->raw())), reinterpret_cast<int*>(encoder_layers_6_self_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_94_0,
        encoder_layers_6_self_attn_proj_weight,
        encoder_layers_6_self_attn_proj_bias,
        reshape_90_0,

        reshape_97_0,
        global_workspace,

     1,


        &reshape_95_0_dim_0,

        &reshape_95_0_dim_1,


        &encoder_layers_6_self_attn_proj_weight_dim_0,

        &encoder_layers_6_self_attn_proj_weight_dim_1,


        &reshape_95_0_dim_0,

        &encoder_layers_6_self_attn_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_97_0_dim_0;

        M *= reshape_97_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_97_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_98_0->raw())), reinterpret_cast<half*>(&(reshape_97_0->raw())), reinterpret_cast<half*>(&(encoder_layers_6_layer_norm2_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_6_layer_norm2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_98_0,
        encoder_layers_6_mlp_fc1_weight,

        encoder_layers_6_mlp_fc1_bias,

        gemm_rcr_bias_99_0,
        global_workspace,
        1,

        &reshape_97_0_dim_0,

        &reshape_97_0_dim_1,

        &reshape_97_0_dim_2,


        &encoder_layers_6_mlp_fc1_weight_dim_0,

        &encoder_layers_6_mlp_fc1_weight_dim_1,


        &reshape_97_0_dim_0,

        &reshape_97_0_dim_1,

        &encoder_layers_6_mlp_fc1_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_183_n_elements = 1 * 64 * 3072;
        invoke_fused_elementwise_183(reinterpret_cast<half*>(elementwise_102_0), reinterpret_cast<half*>(gemm_rcr_bias_99_0),  fused_elementwise_183_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_19(
        elementwise_102_0,
        encoder_layers_6_mlp_fc2_weight,
        encoder_layers_6_mlp_fc2_bias,
        reshape_97_0,

        reshape_104_0,
        global_workspace,

     2,


        &reshape_97_0_dim_0,

        &reshape_97_0_dim_1,

        &encoder_layers_6_mlp_fc1_weight_dim_0,


        &encoder_layers_6_mlp_fc2_weight_dim_0,

        &encoder_layers_6_mlp_fc2_weight_dim_1,


        &reshape_97_0_dim_0,

        &reshape_97_0_dim_1,

        &encoder_layers_6_mlp_fc2_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_104_0_dim_0;

        M *= reshape_104_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_104_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_105_0->raw())), reinterpret_cast<half*>(&(reshape_104_0->raw())), reinterpret_cast<half*>(&(encoder_layers_7_layer_norm1_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_7_layer_norm1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_8(
        layernorm_105_0,
        encoder_layers_7_self_attn_qkv_weight,

        encoder_layers_7_self_attn_qkv_bias,

        reshape_107_0,
        global_workspace,
        1,

        &reshape_104_0_dim_0,

        &reshape_104_0_dim_1,

        &reshape_104_0_dim_2,


        &encoder_layers_7_self_attn_qkv_weight_dim_0,

        &encoder_layers_7_self_attn_qkv_weight_dim_1,


        &reshape_104_0_dim_0,

        &reshape_104_0_dim_1,

        &encoder_layers_7_self_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_108_0->raw())), reinterpret_cast<half*>(&(reshape_107_0->raw())), reinterpret_cast<int*>(encoder_layers_7_self_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_108_0,
        encoder_layers_7_self_attn_proj_weight,
        encoder_layers_7_self_attn_proj_bias,
        reshape_104_0,

        reshape_111_0,
        global_workspace,

     1,


        &reshape_109_0_dim_0,

        &reshape_109_0_dim_1,


        &encoder_layers_7_self_attn_proj_weight_dim_0,

        &encoder_layers_7_self_attn_proj_weight_dim_1,


        &reshape_109_0_dim_0,

        &encoder_layers_7_self_attn_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_111_0_dim_0;

        M *= reshape_111_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_111_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_112_0->raw())), reinterpret_cast<half*>(&(reshape_111_0->raw())), reinterpret_cast<half*>(&(encoder_layers_7_layer_norm2_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_7_layer_norm2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_112_0,
        encoder_layers_7_mlp_fc1_weight,

        encoder_layers_7_mlp_fc1_bias,

        gemm_rcr_bias_113_0,
        global_workspace,
        1,

        &reshape_111_0_dim_0,

        &reshape_111_0_dim_1,

        &reshape_111_0_dim_2,


        &encoder_layers_7_mlp_fc1_weight_dim_0,

        &encoder_layers_7_mlp_fc1_weight_dim_1,


        &reshape_111_0_dim_0,

        &reshape_111_0_dim_1,

        &encoder_layers_7_mlp_fc1_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_184_n_elements = 1 * 64 * 3072;
        invoke_fused_elementwise_184(reinterpret_cast<half*>(elementwise_116_0), reinterpret_cast<half*>(gemm_rcr_bias_113_0),  fused_elementwise_184_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_19(
        elementwise_116_0,
        encoder_layers_7_mlp_fc2_weight,
        encoder_layers_7_mlp_fc2_bias,
        reshape_111_0,

        reshape_118_0,
        global_workspace,

     2,


        &reshape_111_0_dim_0,

        &reshape_111_0_dim_1,

        &encoder_layers_7_mlp_fc1_weight_dim_0,


        &encoder_layers_7_mlp_fc2_weight_dim_0,

        &encoder_layers_7_mlp_fc2_weight_dim_1,


        &reshape_111_0_dim_0,

        &reshape_111_0_dim_1,

        &encoder_layers_7_mlp_fc2_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_118_0_dim_0;

        M *= reshape_118_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_118_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_119_0->raw())), reinterpret_cast<half*>(&(reshape_118_0->raw())), reinterpret_cast<half*>(&(encoder_layers_8_layer_norm1_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_8_layer_norm1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_8(
        layernorm_119_0,
        encoder_layers_8_self_attn_qkv_weight,

        encoder_layers_8_self_attn_qkv_bias,

        reshape_121_0,
        global_workspace,
        1,

        &reshape_118_0_dim_0,

        &reshape_118_0_dim_1,

        &reshape_118_0_dim_2,


        &encoder_layers_8_self_attn_qkv_weight_dim_0,

        &encoder_layers_8_self_attn_qkv_weight_dim_1,


        &reshape_118_0_dim_0,

        &reshape_118_0_dim_1,

        &encoder_layers_8_self_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_122_0->raw())), reinterpret_cast<half*>(&(reshape_121_0->raw())), reinterpret_cast<int*>(encoder_layers_8_self_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_122_0,
        encoder_layers_8_self_attn_proj_weight,
        encoder_layers_8_self_attn_proj_bias,
        reshape_118_0,

        reshape_125_0,
        global_workspace,

     1,


        &reshape_123_0_dim_0,

        &reshape_123_0_dim_1,


        &encoder_layers_8_self_attn_proj_weight_dim_0,

        &encoder_layers_8_self_attn_proj_weight_dim_1,


        &reshape_123_0_dim_0,

        &encoder_layers_8_self_attn_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_125_0_dim_0;

        M *= reshape_125_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_125_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_126_0->raw())), reinterpret_cast<half*>(&(reshape_125_0->raw())), reinterpret_cast<half*>(&(encoder_layers_8_layer_norm2_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_8_layer_norm2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_126_0,
        encoder_layers_8_mlp_fc1_weight,

        encoder_layers_8_mlp_fc1_bias,

        gemm_rcr_bias_127_0,
        global_workspace,
        1,

        &reshape_125_0_dim_0,

        &reshape_125_0_dim_1,

        &reshape_125_0_dim_2,


        &encoder_layers_8_mlp_fc1_weight_dim_0,

        &encoder_layers_8_mlp_fc1_weight_dim_1,


        &reshape_125_0_dim_0,

        &reshape_125_0_dim_1,

        &encoder_layers_8_mlp_fc1_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_185_n_elements = 1 * 64 * 3072;
        invoke_fused_elementwise_185(reinterpret_cast<half*>(elementwise_130_0), reinterpret_cast<half*>(gemm_rcr_bias_127_0),  fused_elementwise_185_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_19(
        elementwise_130_0,
        encoder_layers_8_mlp_fc2_weight,
        encoder_layers_8_mlp_fc2_bias,
        reshape_125_0,

        reshape_132_0,
        global_workspace,

     2,


        &reshape_125_0_dim_0,

        &reshape_125_0_dim_1,

        &encoder_layers_8_mlp_fc1_weight_dim_0,


        &encoder_layers_8_mlp_fc2_weight_dim_0,

        &encoder_layers_8_mlp_fc2_weight_dim_1,


        &reshape_125_0_dim_0,

        &reshape_125_0_dim_1,

        &encoder_layers_8_mlp_fc2_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_132_0_dim_0;

        M *= reshape_132_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_132_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_133_0->raw())), reinterpret_cast<half*>(&(reshape_132_0->raw())), reinterpret_cast<half*>(&(encoder_layers_9_layer_norm1_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_9_layer_norm1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_8(
        layernorm_133_0,
        encoder_layers_9_self_attn_qkv_weight,

        encoder_layers_9_self_attn_qkv_bias,

        reshape_135_0,
        global_workspace,
        1,

        &reshape_132_0_dim_0,

        &reshape_132_0_dim_1,

        &reshape_132_0_dim_2,


        &encoder_layers_9_self_attn_qkv_weight_dim_0,

        &encoder_layers_9_self_attn_qkv_weight_dim_1,


        &reshape_132_0_dim_0,

        &reshape_132_0_dim_1,

        &encoder_layers_9_self_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_136_0->raw())), reinterpret_cast<half*>(&(reshape_135_0->raw())), reinterpret_cast<int*>(encoder_layers_9_self_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_136_0,
        encoder_layers_9_self_attn_proj_weight,
        encoder_layers_9_self_attn_proj_bias,
        reshape_132_0,

        reshape_139_0,
        global_workspace,

     1,


        &reshape_137_0_dim_0,

        &reshape_137_0_dim_1,


        &encoder_layers_9_self_attn_proj_weight_dim_0,

        &encoder_layers_9_self_attn_proj_weight_dim_1,


        &reshape_137_0_dim_0,

        &encoder_layers_9_self_attn_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_139_0_dim_0;

        M *= reshape_139_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_139_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_140_0->raw())), reinterpret_cast<half*>(&(reshape_139_0->raw())), reinterpret_cast<half*>(&(encoder_layers_9_layer_norm2_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_9_layer_norm2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_140_0,
        encoder_layers_9_mlp_fc1_weight,

        encoder_layers_9_mlp_fc1_bias,

        gemm_rcr_bias_141_0,
        global_workspace,
        1,

        &reshape_139_0_dim_0,

        &reshape_139_0_dim_1,

        &reshape_139_0_dim_2,


        &encoder_layers_9_mlp_fc1_weight_dim_0,

        &encoder_layers_9_mlp_fc1_weight_dim_1,


        &reshape_139_0_dim_0,

        &reshape_139_0_dim_1,

        &encoder_layers_9_mlp_fc1_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_186_n_elements = 1 * 64 * 3072;
        invoke_fused_elementwise_186(reinterpret_cast<half*>(elementwise_144_0), reinterpret_cast<half*>(gemm_rcr_bias_141_0),  fused_elementwise_186_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_19(
        elementwise_144_0,
        encoder_layers_9_mlp_fc2_weight,
        encoder_layers_9_mlp_fc2_bias,
        reshape_139_0,

        reshape_146_0,
        global_workspace,

     2,


        &reshape_139_0_dim_0,

        &reshape_139_0_dim_1,

        &encoder_layers_9_mlp_fc1_weight_dim_0,


        &encoder_layers_9_mlp_fc2_weight_dim_0,

        &encoder_layers_9_mlp_fc2_weight_dim_1,


        &reshape_139_0_dim_0,

        &reshape_139_0_dim_1,

        &encoder_layers_9_mlp_fc2_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_146_0_dim_0;

        M *= reshape_146_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_146_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_147_0->raw())), reinterpret_cast<half*>(&(reshape_146_0->raw())), reinterpret_cast<half*>(&(encoder_layers_10_layer_norm1_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_10_layer_norm1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_8(
        layernorm_147_0,
        encoder_layers_10_self_attn_qkv_weight,

        encoder_layers_10_self_attn_qkv_bias,

        reshape_149_0,
        global_workspace,
        1,

        &reshape_146_0_dim_0,

        &reshape_146_0_dim_1,

        &reshape_146_0_dim_2,


        &encoder_layers_10_self_attn_qkv_weight_dim_0,

        &encoder_layers_10_self_attn_qkv_weight_dim_1,


        &reshape_146_0_dim_0,

        &reshape_146_0_dim_1,

        &encoder_layers_10_self_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_150_0->raw())), reinterpret_cast<half*>(&(reshape_149_0->raw())), reinterpret_cast<int*>(encoder_layers_10_self_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_150_0,
        encoder_layers_10_self_attn_proj_weight,
        encoder_layers_10_self_attn_proj_bias,
        reshape_146_0,

        reshape_153_0,
        global_workspace,

     1,


        &reshape_151_0_dim_0,

        &reshape_151_0_dim_1,


        &encoder_layers_10_self_attn_proj_weight_dim_0,

        &encoder_layers_10_self_attn_proj_weight_dim_1,


        &reshape_151_0_dim_0,

        &encoder_layers_10_self_attn_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_153_0_dim_0;

        M *= reshape_153_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_153_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_154_0->raw())), reinterpret_cast<half*>(&(reshape_153_0->raw())), reinterpret_cast<half*>(&(encoder_layers_10_layer_norm2_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_10_layer_norm2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_154_0,
        encoder_layers_10_mlp_fc1_weight,

        encoder_layers_10_mlp_fc1_bias,

        gemm_rcr_bias_155_0,
        global_workspace,
        1,

        &reshape_153_0_dim_0,

        &reshape_153_0_dim_1,

        &reshape_153_0_dim_2,


        &encoder_layers_10_mlp_fc1_weight_dim_0,

        &encoder_layers_10_mlp_fc1_weight_dim_1,


        &reshape_153_0_dim_0,

        &reshape_153_0_dim_1,

        &encoder_layers_10_mlp_fc1_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_187_n_elements = 1 * 64 * 3072;
        invoke_fused_elementwise_187(reinterpret_cast<half*>(elementwise_158_0), reinterpret_cast<half*>(gemm_rcr_bias_155_0),  fused_elementwise_187_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_19(
        elementwise_158_0,
        encoder_layers_10_mlp_fc2_weight,
        encoder_layers_10_mlp_fc2_bias,
        reshape_153_0,

        reshape_160_0,
        global_workspace,

     2,


        &reshape_153_0_dim_0,

        &reshape_153_0_dim_1,

        &encoder_layers_10_mlp_fc1_weight_dim_0,


        &encoder_layers_10_mlp_fc2_weight_dim_0,

        &encoder_layers_10_mlp_fc2_weight_dim_1,


        &reshape_153_0_dim_0,

        &reshape_153_0_dim_1,

        &encoder_layers_10_mlp_fc2_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_160_0_dim_0;

        M *= reshape_160_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_160_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_161_0->raw())), reinterpret_cast<half*>(&(reshape_160_0->raw())), reinterpret_cast<half*>(&(encoder_layers_11_layer_norm1_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_11_layer_norm1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_8(
        layernorm_161_0,
        encoder_layers_11_self_attn_qkv_weight,

        encoder_layers_11_self_attn_qkv_bias,

        reshape_163_0,
        global_workspace,
        1,

        &reshape_160_0_dim_0,

        &reshape_160_0_dim_1,

        &reshape_160_0_dim_2,


        &encoder_layers_11_self_attn_qkv_weight_dim_0,

        &encoder_layers_11_self_attn_qkv_weight_dim_1,


        &reshape_160_0_dim_0,

        &reshape_160_0_dim_1,

        &encoder_layers_11_self_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_164_0->raw())), reinterpret_cast<half*>(&(reshape_163_0->raw())), reinterpret_cast<int*>(encoder_layers_11_self_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1536 * sizeof(float)),
        1,
        128,
        12,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_164_0,
        encoder_layers_11_self_attn_proj_weight,
        encoder_layers_11_self_attn_proj_bias,
        reshape_160_0,

        reshape_167_0,
        global_workspace,

     1,


        &reshape_165_0_dim_0,

        &reshape_165_0_dim_1,


        &encoder_layers_11_self_attn_proj_weight_dim_0,

        &encoder_layers_11_self_attn_proj_weight_dim_1,


        &reshape_165_0_dim_0,

        &encoder_layers_11_self_attn_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_167_0_dim_0;

        M *= reshape_167_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_167_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(layernorm_168_0->raw())), reinterpret_cast<half*>(&(reshape_167_0->raw())), reinterpret_cast<half*>(&(encoder_layers_11_layer_norm2_weight->raw())), reinterpret_cast<half*>(&(encoder_layers_11_layer_norm2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_168_0,
        encoder_layers_11_mlp_fc1_weight,

        encoder_layers_11_mlp_fc1_bias,

        gemm_rcr_bias_169_0,
        global_workspace,
        1,

        &reshape_167_0_dim_0,

        &reshape_167_0_dim_1,

        &reshape_167_0_dim_2,


        &encoder_layers_11_mlp_fc1_weight_dim_0,

        &encoder_layers_11_mlp_fc1_weight_dim_1,


        &reshape_167_0_dim_0,

        &reshape_167_0_dim_1,

        &encoder_layers_11_mlp_fc1_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_188_n_elements = 1 * 64 * 3072;
        invoke_fused_elementwise_188(reinterpret_cast<half*>(elementwise_172_0), reinterpret_cast<half*>(gemm_rcr_bias_169_0),  fused_elementwise_188_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_19(
        elementwise_172_0,
        encoder_layers_11_mlp_fc2_weight,
        encoder_layers_11_mlp_fc2_bias,
        reshape_167_0,

        reshape_174_0,
        global_workspace,

     2,


        &reshape_167_0_dim_0,

        &reshape_167_0_dim_1,

        &encoder_layers_11_mlp_fc1_weight_dim_0,


        &encoder_layers_11_mlp_fc2_weight_dim_0,

        &encoder_layers_11_mlp_fc2_weight_dim_1,


        &reshape_167_0_dim_0,

        &reshape_167_0_dim_1,

        &encoder_layers_11_mlp_fc2_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_174_0_dim_0;

        M *= reshape_174_0_dim_1;

    

        int64_t N = 1;

        N *= reshape_174_0_dim_2;

    
        layernorm_7(
           reinterpret_cast<half*>(&(output_0->raw())), reinterpret_cast<half*>(&(reshape_174_0->raw())), reinterpret_cast<half*>(&(final_layer_norm_weight->raw())), reinterpret_cast<half*>(&(final_layer_norm_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
      DeviceToDeviceCopies(stream);
    }

    bool IsPending() {
      auto query = QueryEvent(run_finished);
      if (query == GetDeviceNotReady()) {
        return true;
      }
      if (query != GetDeviceSuccess()) {
        LOG(WARNING) << "Pending model run did not finish successfully. Error: "
                    << GetErrorString(query);
      }
      return false;
    }

    void WaitForCompletion() {
      DEVICE_CHECK(EventSynchronize(run_finished));
    }

    size_t NumInputs() const {
      return num_inputs;
    }

    size_t NumOutputs() const {
      return params.size() - num_inputs;
    }

    void SetParam(const void* src, size_t param_idx) {
      CHECK_VECTOR_ACCESS(params, param_idx)
      // const_cast is not ideal here, but it is unfortunately
      // necessary:
      // 1) We store outputs and inputs in the same vector,
      //    and outputs cannot be const.
      // 2) Most of the codegen is not const-correct (most ops
      //    require non-const pointers). So even if we put const
      //    pointers into params, a const_cast would be required
      //    somewhere else.
      params[param_idx].ptr = const_cast<void*>(src);
    }

    void SetInput(const void* src, const AITemplateParamShape& shape, size_t idx) {
      SetInputShape(shape, idx);
      SetParam(src, idx);
    }

    void SetOutput(void* src, size_t idx) {
      SetParam(src, idx + num_inputs);
    }

    // Write the (possibly dynamic) output shape to the given pointer.
    // Note that this should be called _after_ the shape inference in
    // Run() is finished. output_shape_out should be able to store
    // at least GetOutputMaximumShape(idx).size values.
    void GetOutputShape(size_t idx, int64_t* output_shape_out) {
      const auto param_idx = idx + num_inputs;
      CHECK_VECTOR_ACCESS(params, param_idx);
      const auto& shape_ptrs = params[param_idx].shape_ptrs;
      for (size_t i = 0; i < shape_ptrs.size(); ++i) {
        output_shape_out[i] = shape_ptrs[i].GetValue();
      }
    }

    void SetConstant(const char* name, const void* src) {
      auto it = constant_name_to_ptr_.find(name);
      if (it == constant_name_to_ptr_.end()) {
        throw std::out_of_range(std::string("Could not find constant ") + name);
      }
      const void** ptr = it->second;
      *ptr = src;
    }

  private:
    void SetInputShape(const AITemplateParamShape& shape, size_t idx) {
      auto& param = params[idx];
      if (shape.size != param.shape_ptrs.size()) {
        throw std::runtime_error(
          "[SetInputShape] Got wrong param shape for input " + std::to_string(idx) +
          "; expected " + std::to_string(param.shape_ptrs.size()) + ", got " +
          std::to_string(shape.size));
      }
      for (size_t i = 0; i < param.shape_ptrs.size(); ++i) {
        param.shape_ptrs[i].SetValue(shape.shape_data[i]);
      }
    }

    void RunAsGraph(StreamType stream) {
      DEVICE_CHECK(StreamBeginCapture(graph_capture_stream));
      try {
        RunImpl(graph_capture_stream);
      } catch (...) {
        DEVICE_CHECK(StreamEndCapture(graph_capture_stream, &graph));
        throw;
      }
      DEVICE_CHECK(StreamEndCapture(graph_capture_stream, &graph));

      if (graph_exec == nullptr) {
        DEVICE_CHECK(GraphInstantiate(&graph_exec, graph));
      } else if (GraphExecUpdate(graph_exec, graph) != GetDeviceSuccess()) {
        DEVICE_CHECK(GraphExecDestroy(graph_exec));
        DEVICE_CHECK(GraphInstantiate(&graph_exec, graph));
      }

      DEVICE_CHECK(GraphExecLaunch(graph_exec, stream));
    }

    int device_idx;
    int max_smem_size{0};
    DevicePropertyType device_properties;
    // This event tracks when the inference is finished
    // so that this Model may be reclaimed by its owning
    // ModelContainer.
    EventType run_finished;
    // A blob of memory used for storing intermediate tensors.
    GPUPtr blob;
    // Memory for constants that were folded into the *.so. Unowned by Model,
    // owned by ModelContainer.
    // TODO: make this const. It can't be const right now because we derive
    // tensor pointers from it, and no tensor pointers are const.
    uint8_t* constants;
    size_t num_inputs;

    // The workspace blob is used as scratch memory. See
    // _generate_workspace in memory planning for more information.
    GPUPtr workspace;
    uint8_t* global_workspace{nullptr};
    uint8_t* unique_workspace{nullptr};

    class ParamDim {
      public:
        ParamDim(int64_t lower_bound, int64_t upper_bound, int64_t* value) :
          lower_bound_(lower_bound),
          upper_bound_(upper_bound),
          value_(value) {}

        void SetValue(int64_t new_value) {
          if (new_value < lower_bound_ || new_value > upper_bound_) {
            throw std::out_of_range(
              "[SetValue] Dimension got value out of bounds; expected value to be in [" +
              std::to_string(lower_bound_) + ", " + std::to_string(upper_bound_) + "], but got " +
              std::to_string(new_value)
            );
          }
          *value_ = new_value;
        }

        int64_t GetValue() const {
          return *value_;
        }

      private:
        int64_t lower_bound_;
        int64_t upper_bound_;
        int64_t* value_;
    };

    struct ParamInfo {
      void* ptr = nullptr;
      // TODO add offset
      const char* name;
      std::vector<ParamDim> shape_ptrs;
    };

    // Contains info for all tensors marked as inputs
    // or outputs. The first num_inputs elements are the inputs.
    // Constants are not included.
    std::vector<ParamInfo> params;

    GraphExecType graph_exec = nullptr;
    GraphType graph = nullptr;
    StreamType graph_capture_stream;

    std::unordered_map<std::string, const void**> constant_name_to_ptr_;

    constexpr static bool target_has_graph_mode = true;

   int64_t* input0 {nullptr};
   int64_t* reshape_0_0 {nullptr};
   cutlass::half_t* embeddings_token_embedding_weight {nullptr};
   cutlass::half_t* batch_gather_1_0 {nullptr};
   int64_t* input1 {nullptr};
   int64_t* reshape_2_0 {nullptr};
   cutlass::half_t* embeddings_position_embedding_weight {nullptr};
   cutlass::half_t* batch_gather_3_0 {nullptr};
   cutlass::half_t* size_5_0 {nullptr};
   cutlass::half_t* size_5_1 {nullptr};
   cutlass::half_t* reshape_6_0 {nullptr};
   cutlass::half_t* encoder_layers_0_layer_norm1_weight {nullptr};
   cutlass::half_t* encoder_layers_0_layer_norm1_bias {nullptr};
   cutlass::half_t* layernorm_7_0 {nullptr};
   cutlass::half_t* encoder_layers_0_self_attn_qkv_weight {nullptr};
   cutlass::half_t* encoder_layers_0_self_attn_qkv_bias {nullptr};
   cutlass::half_t* reshape_9_0 {nullptr};
   int32_t* encoder_layers_0_self_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_10_0 {nullptr};
   cutlass::half_t* encoder_layers_0_self_attn_proj_weight {nullptr};
   cutlass::half_t* encoder_layers_0_self_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_13_0 {nullptr};
   cutlass::half_t* encoder_layers_0_layer_norm2_weight {nullptr};
   cutlass::half_t* encoder_layers_0_layer_norm2_bias {nullptr};
   cutlass::half_t* layernorm_14_0 {nullptr};
   cutlass::half_t* encoder_layers_0_mlp_fc1_weight {nullptr};
   cutlass::half_t* encoder_layers_0_mlp_fc1_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_15_0 {nullptr};
   cutlass::half_t* elementwise_18_0 {nullptr};
   cutlass::half_t* encoder_layers_0_mlp_fc2_weight {nullptr};
   cutlass::half_t* encoder_layers_0_mlp_fc2_bias {nullptr};
   cutlass::half_t* reshape_20_0 {nullptr};
   cutlass::half_t* encoder_layers_1_layer_norm1_weight {nullptr};
   cutlass::half_t* encoder_layers_1_layer_norm1_bias {nullptr};
   cutlass::half_t* layernorm_21_0 {nullptr};
   cutlass::half_t* encoder_layers_1_self_attn_qkv_weight {nullptr};
   cutlass::half_t* encoder_layers_1_self_attn_qkv_bias {nullptr};
   cutlass::half_t* reshape_23_0 {nullptr};
   int32_t* encoder_layers_1_self_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_24_0 {nullptr};
   cutlass::half_t* encoder_layers_1_self_attn_proj_weight {nullptr};
   cutlass::half_t* encoder_layers_1_self_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_27_0 {nullptr};
   cutlass::half_t* encoder_layers_1_layer_norm2_weight {nullptr};
   cutlass::half_t* encoder_layers_1_layer_norm2_bias {nullptr};
   cutlass::half_t* layernorm_28_0 {nullptr};
   cutlass::half_t* encoder_layers_1_mlp_fc1_weight {nullptr};
   cutlass::half_t* encoder_layers_1_mlp_fc1_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_29_0 {nullptr};
   cutlass::half_t* elementwise_32_0 {nullptr};
   cutlass::half_t* encoder_layers_1_mlp_fc2_weight {nullptr};
   cutlass::half_t* encoder_layers_1_mlp_fc2_bias {nullptr};
   cutlass::half_t* reshape_34_0 {nullptr};
   cutlass::half_t* encoder_layers_2_layer_norm1_weight {nullptr};
   cutlass::half_t* encoder_layers_2_layer_norm1_bias {nullptr};
   cutlass::half_t* layernorm_35_0 {nullptr};
   cutlass::half_t* encoder_layers_2_self_attn_qkv_weight {nullptr};
   cutlass::half_t* encoder_layers_2_self_attn_qkv_bias {nullptr};
   cutlass::half_t* reshape_37_0 {nullptr};
   int32_t* encoder_layers_2_self_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_38_0 {nullptr};
   cutlass::half_t* encoder_layers_2_self_attn_proj_weight {nullptr};
   cutlass::half_t* encoder_layers_2_self_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_41_0 {nullptr};
   cutlass::half_t* encoder_layers_2_layer_norm2_weight {nullptr};
   cutlass::half_t* encoder_layers_2_layer_norm2_bias {nullptr};
   cutlass::half_t* layernorm_42_0 {nullptr};
   cutlass::half_t* encoder_layers_2_mlp_fc1_weight {nullptr};
   cutlass::half_t* encoder_layers_2_mlp_fc1_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_43_0 {nullptr};
   cutlass::half_t* elementwise_46_0 {nullptr};
   cutlass::half_t* encoder_layers_2_mlp_fc2_weight {nullptr};
   cutlass::half_t* encoder_layers_2_mlp_fc2_bias {nullptr};
   cutlass::half_t* reshape_48_0 {nullptr};
   cutlass::half_t* encoder_layers_3_layer_norm1_weight {nullptr};
   cutlass::half_t* encoder_layers_3_layer_norm1_bias {nullptr};
   cutlass::half_t* layernorm_49_0 {nullptr};
   cutlass::half_t* encoder_layers_3_self_attn_qkv_weight {nullptr};
   cutlass::half_t* encoder_layers_3_self_attn_qkv_bias {nullptr};
   cutlass::half_t* reshape_51_0 {nullptr};
   int32_t* encoder_layers_3_self_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_52_0 {nullptr};
   cutlass::half_t* encoder_layers_3_self_attn_proj_weight {nullptr};
   cutlass::half_t* encoder_layers_3_self_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_55_0 {nullptr};
   cutlass::half_t* encoder_layers_3_layer_norm2_weight {nullptr};
   cutlass::half_t* encoder_layers_3_layer_norm2_bias {nullptr};
   cutlass::half_t* layernorm_56_0 {nullptr};
   cutlass::half_t* encoder_layers_3_mlp_fc1_weight {nullptr};
   cutlass::half_t* encoder_layers_3_mlp_fc1_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_57_0 {nullptr};
   cutlass::half_t* elementwise_60_0 {nullptr};
   cutlass::half_t* encoder_layers_3_mlp_fc2_weight {nullptr};
   cutlass::half_t* encoder_layers_3_mlp_fc2_bias {nullptr};
   cutlass::half_t* reshape_62_0 {nullptr};
   cutlass::half_t* encoder_layers_4_layer_norm1_weight {nullptr};
   cutlass::half_t* encoder_layers_4_layer_norm1_bias {nullptr};
   cutlass::half_t* layernorm_63_0 {nullptr};
   cutlass::half_t* encoder_layers_4_self_attn_qkv_weight {nullptr};
   cutlass::half_t* encoder_layers_4_self_attn_qkv_bias {nullptr};
   cutlass::half_t* reshape_65_0 {nullptr};
   int32_t* encoder_layers_4_self_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_66_0 {nullptr};
   cutlass::half_t* encoder_layers_4_self_attn_proj_weight {nullptr};
   cutlass::half_t* encoder_layers_4_self_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_69_0 {nullptr};
   cutlass::half_t* encoder_layers_4_layer_norm2_weight {nullptr};
   cutlass::half_t* encoder_layers_4_layer_norm2_bias {nullptr};
   cutlass::half_t* layernorm_70_0 {nullptr};
   cutlass::half_t* encoder_layers_4_mlp_fc1_weight {nullptr};
   cutlass::half_t* encoder_layers_4_mlp_fc1_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_71_0 {nullptr};
   cutlass::half_t* elementwise_74_0 {nullptr};
   cutlass::half_t* encoder_layers_4_mlp_fc2_weight {nullptr};
   cutlass::half_t* encoder_layers_4_mlp_fc2_bias {nullptr};
   cutlass::half_t* reshape_76_0 {nullptr};
   cutlass::half_t* encoder_layers_5_layer_norm1_weight {nullptr};
   cutlass::half_t* encoder_layers_5_layer_norm1_bias {nullptr};
   cutlass::half_t* layernorm_77_0 {nullptr};
   cutlass::half_t* encoder_layers_5_self_attn_qkv_weight {nullptr};
   cutlass::half_t* encoder_layers_5_self_attn_qkv_bias {nullptr};
   cutlass::half_t* reshape_79_0 {nullptr};
   int32_t* encoder_layers_5_self_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_80_0 {nullptr};
   cutlass::half_t* encoder_layers_5_self_attn_proj_weight {nullptr};
   cutlass::half_t* encoder_layers_5_self_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_83_0 {nullptr};
   cutlass::half_t* encoder_layers_5_layer_norm2_weight {nullptr};
   cutlass::half_t* encoder_layers_5_layer_norm2_bias {nullptr};
   cutlass::half_t* layernorm_84_0 {nullptr};
   cutlass::half_t* encoder_layers_5_mlp_fc1_weight {nullptr};
   cutlass::half_t* encoder_layers_5_mlp_fc1_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_85_0 {nullptr};
   cutlass::half_t* elementwise_88_0 {nullptr};
   cutlass::half_t* encoder_layers_5_mlp_fc2_weight {nullptr};
   cutlass::half_t* encoder_layers_5_mlp_fc2_bias {nullptr};
   cutlass::half_t* reshape_90_0 {nullptr};
   cutlass::half_t* encoder_layers_6_layer_norm1_weight {nullptr};
   cutlass::half_t* encoder_layers_6_layer_norm1_bias {nullptr};
   cutlass::half_t* layernorm_91_0 {nullptr};
   cutlass::half_t* encoder_layers_6_self_attn_qkv_weight {nullptr};
   cutlass::half_t* encoder_layers_6_self_attn_qkv_bias {nullptr};
   cutlass::half_t* reshape_93_0 {nullptr};
   int32_t* encoder_layers_6_self_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_94_0 {nullptr};
   cutlass::half_t* encoder_layers_6_self_attn_proj_weight {nullptr};
   cutlass::half_t* encoder_layers_6_self_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_97_0 {nullptr};
   cutlass::half_t* encoder_layers_6_layer_norm2_weight {nullptr};
   cutlass::half_t* encoder_layers_6_layer_norm2_bias {nullptr};
   cutlass::half_t* layernorm_98_0 {nullptr};
   cutlass::half_t* encoder_layers_6_mlp_fc1_weight {nullptr};
   cutlass::half_t* encoder_layers_6_mlp_fc1_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_99_0 {nullptr};
   cutlass::half_t* elementwise_102_0 {nullptr};
   cutlass::half_t* encoder_layers_6_mlp_fc2_weight {nullptr};
   cutlass::half_t* encoder_layers_6_mlp_fc2_bias {nullptr};
   cutlass::half_t* reshape_104_0 {nullptr};
   cutlass::half_t* encoder_layers_7_layer_norm1_weight {nullptr};
   cutlass::half_t* encoder_layers_7_layer_norm1_bias {nullptr};
   cutlass::half_t* layernorm_105_0 {nullptr};
   cutlass::half_t* encoder_layers_7_self_attn_qkv_weight {nullptr};
   cutlass::half_t* encoder_layers_7_self_attn_qkv_bias {nullptr};
   cutlass::half_t* reshape_107_0 {nullptr};
   int32_t* encoder_layers_7_self_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_108_0 {nullptr};
   cutlass::half_t* encoder_layers_7_self_attn_proj_weight {nullptr};
   cutlass::half_t* encoder_layers_7_self_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_111_0 {nullptr};
   cutlass::half_t* encoder_layers_7_layer_norm2_weight {nullptr};
   cutlass::half_t* encoder_layers_7_layer_norm2_bias {nullptr};
   cutlass::half_t* layernorm_112_0 {nullptr};
   cutlass::half_t* encoder_layers_7_mlp_fc1_weight {nullptr};
   cutlass::half_t* encoder_layers_7_mlp_fc1_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_113_0 {nullptr};
   cutlass::half_t* elementwise_116_0 {nullptr};
   cutlass::half_t* encoder_layers_7_mlp_fc2_weight {nullptr};
   cutlass::half_t* encoder_layers_7_mlp_fc2_bias {nullptr};
   cutlass::half_t* reshape_118_0 {nullptr};
   cutlass::half_t* encoder_layers_8_layer_norm1_weight {nullptr};
   cutlass::half_t* encoder_layers_8_layer_norm1_bias {nullptr};
   cutlass::half_t* layernorm_119_0 {nullptr};
   cutlass::half_t* encoder_layers_8_self_attn_qkv_weight {nullptr};
   cutlass::half_t* encoder_layers_8_self_attn_qkv_bias {nullptr};
   cutlass::half_t* reshape_121_0 {nullptr};
   int32_t* encoder_layers_8_self_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_122_0 {nullptr};
   cutlass::half_t* encoder_layers_8_self_attn_proj_weight {nullptr};
   cutlass::half_t* encoder_layers_8_self_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_125_0 {nullptr};
   cutlass::half_t* encoder_layers_8_layer_norm2_weight {nullptr};
   cutlass::half_t* encoder_layers_8_layer_norm2_bias {nullptr};
   cutlass::half_t* layernorm_126_0 {nullptr};
   cutlass::half_t* encoder_layers_8_mlp_fc1_weight {nullptr};
   cutlass::half_t* encoder_layers_8_mlp_fc1_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_127_0 {nullptr};
   cutlass::half_t* elementwise_130_0 {nullptr};
   cutlass::half_t* encoder_layers_8_mlp_fc2_weight {nullptr};
   cutlass::half_t* encoder_layers_8_mlp_fc2_bias {nullptr};
   cutlass::half_t* reshape_132_0 {nullptr};
   cutlass::half_t* encoder_layers_9_layer_norm1_weight {nullptr};
   cutlass::half_t* encoder_layers_9_layer_norm1_bias {nullptr};
   cutlass::half_t* layernorm_133_0 {nullptr};
   cutlass::half_t* encoder_layers_9_self_attn_qkv_weight {nullptr};
   cutlass::half_t* encoder_layers_9_self_attn_qkv_bias {nullptr};
   cutlass::half_t* reshape_135_0 {nullptr};
   int32_t* encoder_layers_9_self_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_136_0 {nullptr};
   cutlass::half_t* encoder_layers_9_self_attn_proj_weight {nullptr};
   cutlass::half_t* encoder_layers_9_self_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_139_0 {nullptr};
   cutlass::half_t* encoder_layers_9_layer_norm2_weight {nullptr};
   cutlass::half_t* encoder_layers_9_layer_norm2_bias {nullptr};
   cutlass::half_t* layernorm_140_0 {nullptr};
   cutlass::half_t* encoder_layers_9_mlp_fc1_weight {nullptr};
   cutlass::half_t* encoder_layers_9_mlp_fc1_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_141_0 {nullptr};
   cutlass::half_t* elementwise_144_0 {nullptr};
   cutlass::half_t* encoder_layers_9_mlp_fc2_weight {nullptr};
   cutlass::half_t* encoder_layers_9_mlp_fc2_bias {nullptr};
   cutlass::half_t* reshape_146_0 {nullptr};
   cutlass::half_t* encoder_layers_10_layer_norm1_weight {nullptr};
   cutlass::half_t* encoder_layers_10_layer_norm1_bias {nullptr};
   cutlass::half_t* layernorm_147_0 {nullptr};
   cutlass::half_t* encoder_layers_10_self_attn_qkv_weight {nullptr};
   cutlass::half_t* encoder_layers_10_self_attn_qkv_bias {nullptr};
   cutlass::half_t* reshape_149_0 {nullptr};
   int32_t* encoder_layers_10_self_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_150_0 {nullptr};
   cutlass::half_t* encoder_layers_10_self_attn_proj_weight {nullptr};
   cutlass::half_t* encoder_layers_10_self_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_153_0 {nullptr};
   cutlass::half_t* encoder_layers_10_layer_norm2_weight {nullptr};
   cutlass::half_t* encoder_layers_10_layer_norm2_bias {nullptr};
   cutlass::half_t* layernorm_154_0 {nullptr};
   cutlass::half_t* encoder_layers_10_mlp_fc1_weight {nullptr};
   cutlass::half_t* encoder_layers_10_mlp_fc1_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_155_0 {nullptr};
   cutlass::half_t* elementwise_158_0 {nullptr};
   cutlass::half_t* encoder_layers_10_mlp_fc2_weight {nullptr};
   cutlass::half_t* encoder_layers_10_mlp_fc2_bias {nullptr};
   cutlass::half_t* reshape_160_0 {nullptr};
   cutlass::half_t* encoder_layers_11_layer_norm1_weight {nullptr};
   cutlass::half_t* encoder_layers_11_layer_norm1_bias {nullptr};
   cutlass::half_t* layernorm_161_0 {nullptr};
   cutlass::half_t* encoder_layers_11_self_attn_qkv_weight {nullptr};
   cutlass::half_t* encoder_layers_11_self_attn_qkv_bias {nullptr};
   cutlass::half_t* reshape_163_0 {nullptr};
   int32_t* encoder_layers_11_self_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_164_0 {nullptr};
   cutlass::half_t* encoder_layers_11_self_attn_proj_weight {nullptr};
   cutlass::half_t* encoder_layers_11_self_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_167_0 {nullptr};
   cutlass::half_t* encoder_layers_11_layer_norm2_weight {nullptr};
   cutlass::half_t* encoder_layers_11_layer_norm2_bias {nullptr};
   cutlass::half_t* layernorm_168_0 {nullptr};
   cutlass::half_t* encoder_layers_11_mlp_fc1_weight {nullptr};
   cutlass::half_t* encoder_layers_11_mlp_fc1_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_169_0 {nullptr};
   cutlass::half_t* elementwise_172_0 {nullptr};
   cutlass::half_t* encoder_layers_11_mlp_fc2_weight {nullptr};
   cutlass::half_t* encoder_layers_11_mlp_fc2_bias {nullptr};
   cutlass::half_t* reshape_174_0 {nullptr};
   cutlass::half_t* final_layer_norm_weight {nullptr};
   cutlass::half_t* final_layer_norm_bias {nullptr};
   cutlass::half_t* output_0 {nullptr};
   int64_t input0_dim_0 { 1 };
   int64_t input0_dim_1 { 64 };
   int64_t reshape_0_0_dim_0 { 64 };
   int64_t embeddings_token_embedding_weight_dim_0 { 49408 };
   int64_t embeddings_token_embedding_weight_dim_1 { 768 };
   int64_t batch_gather_1_0_dim_0 { 64 };
   int64_t batch_gather_1_0_dim_1 { 768 };
   int64_t input1_dim_0 { 1 };
   int64_t input1_dim_1 { 64 };
   int64_t reshape_2_0_dim_0 { 64 };
   int64_t embeddings_position_embedding_weight_dim_0 { 77 };
   int64_t embeddings_position_embedding_weight_dim_1 { 768 };
   int64_t batch_gather_3_0_dim_0 { 64 };
   int64_t batch_gather_3_0_dim_1 { 768 };
   int64_t reshape_6_0_dim_0 { 1 };
   int64_t reshape_6_0_dim_1 { 64 };
   int64_t reshape_6_0_dim_2 { 768 };
   int64_t encoder_layers_0_layer_norm1_weight_dim_0 { 768 };
   int64_t encoder_layers_0_layer_norm1_bias_dim_0 { 768 };
   int64_t encoder_layers_0_self_attn_qkv_weight_dim_0 { 2304 };
   int64_t encoder_layers_0_self_attn_qkv_weight_dim_1 { 768 };
   int64_t encoder_layers_0_self_attn_qkv_bias_dim_0 { 2304 };
   int64_t reshape_9_0_dim_0 { 64 };
   int64_t reshape_9_0_dim_1 { 3 };
   int64_t reshape_9_0_dim_2 { 12 };
   int64_t reshape_9_0_dim_3 { 64 };
   int64_t encoder_layers_0_self_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_10_0_dim_0 { 64 };
   int64_t flash_attention_10_0_dim_1 { 12 };
   int64_t flash_attention_10_0_dim_2 { 64 };
   int64_t encoder_layers_0_self_attn_proj_weight_dim_0 { 768 };
   int64_t encoder_layers_0_self_attn_proj_weight_dim_1 { 768 };
   int64_t encoder_layers_0_self_attn_proj_bias_dim_0 { 768 };
   int64_t reshape_13_0_dim_0 { 1 };
   int64_t reshape_13_0_dim_1 { 64 };
   int64_t reshape_13_0_dim_2 { 768 };
   int64_t reshape_11_0_dim_0 { 64 };
   int64_t reshape_11_0_dim_1 { 768 };
   int64_t encoder_layers_0_layer_norm2_weight_dim_0 { 768 };
   int64_t encoder_layers_0_layer_norm2_bias_dim_0 { 768 };
   int64_t encoder_layers_0_mlp_fc1_weight_dim_0 { 3072 };
   int64_t encoder_layers_0_mlp_fc1_weight_dim_1 { 768 };
   int64_t encoder_layers_0_mlp_fc1_bias_dim_0 { 3072 };
   int64_t encoder_layers_0_mlp_fc2_weight_dim_0 { 768 };
   int64_t encoder_layers_0_mlp_fc2_weight_dim_1 { 3072 };
   int64_t encoder_layers_0_mlp_fc2_bias_dim_0 { 768 };
   int64_t reshape_20_0_dim_0 { 1 };
   int64_t reshape_20_0_dim_1 { 64 };
   int64_t reshape_20_0_dim_2 { 768 };
   int64_t encoder_layers_1_layer_norm1_weight_dim_0 { 768 };
   int64_t encoder_layers_1_layer_norm1_bias_dim_0 { 768 };
   int64_t encoder_layers_1_self_attn_qkv_weight_dim_0 { 2304 };
   int64_t encoder_layers_1_self_attn_qkv_weight_dim_1 { 768 };
   int64_t encoder_layers_1_self_attn_qkv_bias_dim_0 { 2304 };
   int64_t reshape_23_0_dim_0 { 64 };
   int64_t reshape_23_0_dim_1 { 3 };
   int64_t reshape_23_0_dim_2 { 12 };
   int64_t reshape_23_0_dim_3 { 64 };
   int64_t encoder_layers_1_self_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_24_0_dim_0 { 64 };
   int64_t flash_attention_24_0_dim_1 { 12 };
   int64_t flash_attention_24_0_dim_2 { 64 };
   int64_t encoder_layers_1_self_attn_proj_weight_dim_0 { 768 };
   int64_t encoder_layers_1_self_attn_proj_weight_dim_1 { 768 };
   int64_t encoder_layers_1_self_attn_proj_bias_dim_0 { 768 };
   int64_t reshape_27_0_dim_0 { 1 };
   int64_t reshape_27_0_dim_1 { 64 };
   int64_t reshape_27_0_dim_2 { 768 };
   int64_t reshape_25_0_dim_0 { 64 };
   int64_t reshape_25_0_dim_1 { 768 };
   int64_t encoder_layers_1_layer_norm2_weight_dim_0 { 768 };
   int64_t encoder_layers_1_layer_norm2_bias_dim_0 { 768 };
   int64_t encoder_layers_1_mlp_fc1_weight_dim_0 { 3072 };
   int64_t encoder_layers_1_mlp_fc1_weight_dim_1 { 768 };
   int64_t encoder_layers_1_mlp_fc1_bias_dim_0 { 3072 };
   int64_t encoder_layers_1_mlp_fc2_weight_dim_0 { 768 };
   int64_t encoder_layers_1_mlp_fc2_weight_dim_1 { 3072 };
   int64_t encoder_layers_1_mlp_fc2_bias_dim_0 { 768 };
   int64_t reshape_34_0_dim_0 { 1 };
   int64_t reshape_34_0_dim_1 { 64 };
   int64_t reshape_34_0_dim_2 { 768 };
   int64_t encoder_layers_2_layer_norm1_weight_dim_0 { 768 };
   int64_t encoder_layers_2_layer_norm1_bias_dim_0 { 768 };
   int64_t encoder_layers_2_self_attn_qkv_weight_dim_0 { 2304 };
   int64_t encoder_layers_2_self_attn_qkv_weight_dim_1 { 768 };
   int64_t encoder_layers_2_self_attn_qkv_bias_dim_0 { 2304 };
   int64_t reshape_37_0_dim_0 { 64 };
   int64_t reshape_37_0_dim_1 { 3 };
   int64_t reshape_37_0_dim_2 { 12 };
   int64_t reshape_37_0_dim_3 { 64 };
   int64_t encoder_layers_2_self_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_38_0_dim_0 { 64 };
   int64_t flash_attention_38_0_dim_1 { 12 };
   int64_t flash_attention_38_0_dim_2 { 64 };
   int64_t encoder_layers_2_self_attn_proj_weight_dim_0 { 768 };
   int64_t encoder_layers_2_self_attn_proj_weight_dim_1 { 768 };
   int64_t encoder_layers_2_self_attn_proj_bias_dim_0 { 768 };
   int64_t reshape_41_0_dim_0 { 1 };
   int64_t reshape_41_0_dim_1 { 64 };
   int64_t reshape_41_0_dim_2 { 768 };
   int64_t reshape_39_0_dim_0 { 64 };
   int64_t reshape_39_0_dim_1 { 768 };
   int64_t encoder_layers_2_layer_norm2_weight_dim_0 { 768 };
   int64_t encoder_layers_2_layer_norm2_bias_dim_0 { 768 };
   int64_t encoder_layers_2_mlp_fc1_weight_dim_0 { 3072 };
   int64_t encoder_layers_2_mlp_fc1_weight_dim_1 { 768 };
   int64_t encoder_layers_2_mlp_fc1_bias_dim_0 { 3072 };
   int64_t encoder_layers_2_mlp_fc2_weight_dim_0 { 768 };
   int64_t encoder_layers_2_mlp_fc2_weight_dim_1 { 3072 };
   int64_t encoder_layers_2_mlp_fc2_bias_dim_0 { 768 };
   int64_t reshape_48_0_dim_0 { 1 };
   int64_t reshape_48_0_dim_1 { 64 };
   int64_t reshape_48_0_dim_2 { 768 };
   int64_t encoder_layers_3_layer_norm1_weight_dim_0 { 768 };
   int64_t encoder_layers_3_layer_norm1_bias_dim_0 { 768 };
   int64_t encoder_layers_3_self_attn_qkv_weight_dim_0 { 2304 };
   int64_t encoder_layers_3_self_attn_qkv_weight_dim_1 { 768 };
   int64_t encoder_layers_3_self_attn_qkv_bias_dim_0 { 2304 };
   int64_t reshape_51_0_dim_0 { 64 };
   int64_t reshape_51_0_dim_1 { 3 };
   int64_t reshape_51_0_dim_2 { 12 };
   int64_t reshape_51_0_dim_3 { 64 };
   int64_t encoder_layers_3_self_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_52_0_dim_0 { 64 };
   int64_t flash_attention_52_0_dim_1 { 12 };
   int64_t flash_attention_52_0_dim_2 { 64 };
   int64_t encoder_layers_3_self_attn_proj_weight_dim_0 { 768 };
   int64_t encoder_layers_3_self_attn_proj_weight_dim_1 { 768 };
   int64_t encoder_layers_3_self_attn_proj_bias_dim_0 { 768 };
   int64_t reshape_55_0_dim_0 { 1 };
   int64_t reshape_55_0_dim_1 { 64 };
   int64_t reshape_55_0_dim_2 { 768 };
   int64_t reshape_53_0_dim_0 { 64 };
   int64_t reshape_53_0_dim_1 { 768 };
   int64_t encoder_layers_3_layer_norm2_weight_dim_0 { 768 };
   int64_t encoder_layers_3_layer_norm2_bias_dim_0 { 768 };
   int64_t encoder_layers_3_mlp_fc1_weight_dim_0 { 3072 };
   int64_t encoder_layers_3_mlp_fc1_weight_dim_1 { 768 };
   int64_t encoder_layers_3_mlp_fc1_bias_dim_0 { 3072 };
   int64_t encoder_layers_3_mlp_fc2_weight_dim_0 { 768 };
   int64_t encoder_layers_3_mlp_fc2_weight_dim_1 { 3072 };
   int64_t encoder_layers_3_mlp_fc2_bias_dim_0 { 768 };
   int64_t reshape_62_0_dim_0 { 1 };
   int64_t reshape_62_0_dim_1 { 64 };
   int64_t reshape_62_0_dim_2 { 768 };
   int64_t encoder_layers_4_layer_norm1_weight_dim_0 { 768 };
   int64_t encoder_layers_4_layer_norm1_bias_dim_0 { 768 };
   int64_t encoder_layers_4_self_attn_qkv_weight_dim_0 { 2304 };
   int64_t encoder_layers_4_self_attn_qkv_weight_dim_1 { 768 };
   int64_t encoder_layers_4_self_attn_qkv_bias_dim_0 { 2304 };
   int64_t reshape_65_0_dim_0 { 64 };
   int64_t reshape_65_0_dim_1 { 3 };
   int64_t reshape_65_0_dim_2 { 12 };
   int64_t reshape_65_0_dim_3 { 64 };
   int64_t encoder_layers_4_self_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_66_0_dim_0 { 64 };
   int64_t flash_attention_66_0_dim_1 { 12 };
   int64_t flash_attention_66_0_dim_2 { 64 };
   int64_t encoder_layers_4_self_attn_proj_weight_dim_0 { 768 };
   int64_t encoder_layers_4_self_attn_proj_weight_dim_1 { 768 };
   int64_t encoder_layers_4_self_attn_proj_bias_dim_0 { 768 };
   int64_t reshape_69_0_dim_0 { 1 };
   int64_t reshape_69_0_dim_1 { 64 };
   int64_t reshape_69_0_dim_2 { 768 };
   int64_t reshape_67_0_dim_0 { 64 };
   int64_t reshape_67_0_dim_1 { 768 };
   int64_t encoder_layers_4_layer_norm2_weight_dim_0 { 768 };
   int64_t encoder_layers_4_layer_norm2_bias_dim_0 { 768 };
   int64_t encoder_layers_4_mlp_fc1_weight_dim_0 { 3072 };
   int64_t encoder_layers_4_mlp_fc1_weight_dim_1 { 768 };
   int64_t encoder_layers_4_mlp_fc1_bias_dim_0 { 3072 };
   int64_t encoder_layers_4_mlp_fc2_weight_dim_0 { 768 };
   int64_t encoder_layers_4_mlp_fc2_weight_dim_1 { 3072 };
   int64_t encoder_layers_4_mlp_fc2_bias_dim_0 { 768 };
   int64_t reshape_76_0_dim_0 { 1 };
   int64_t reshape_76_0_dim_1 { 64 };
   int64_t reshape_76_0_dim_2 { 768 };
   int64_t encoder_layers_5_layer_norm1_weight_dim_0 { 768 };
   int64_t encoder_layers_5_layer_norm1_bias_dim_0 { 768 };
   int64_t encoder_layers_5_self_attn_qkv_weight_dim_0 { 2304 };
   int64_t encoder_layers_5_self_attn_qkv_weight_dim_1 { 768 };
   int64_t encoder_layers_5_self_attn_qkv_bias_dim_0 { 2304 };
   int64_t reshape_79_0_dim_0 { 64 };
   int64_t reshape_79_0_dim_1 { 3 };
   int64_t reshape_79_0_dim_2 { 12 };
   int64_t reshape_79_0_dim_3 { 64 };
   int64_t encoder_layers_5_self_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_80_0_dim_0 { 64 };
   int64_t flash_attention_80_0_dim_1 { 12 };
   int64_t flash_attention_80_0_dim_2 { 64 };
   int64_t encoder_layers_5_self_attn_proj_weight_dim_0 { 768 };
   int64_t encoder_layers_5_self_attn_proj_weight_dim_1 { 768 };
   int64_t encoder_layers_5_self_attn_proj_bias_dim_0 { 768 };
   int64_t reshape_83_0_dim_0 { 1 };
   int64_t reshape_83_0_dim_1 { 64 };
   int64_t reshape_83_0_dim_2 { 768 };
   int64_t reshape_81_0_dim_0 { 64 };
   int64_t reshape_81_0_dim_1 { 768 };
   int64_t encoder_layers_5_layer_norm2_weight_dim_0 { 768 };
   int64_t encoder_layers_5_layer_norm2_bias_dim_0 { 768 };
   int64_t encoder_layers_5_mlp_fc1_weight_dim_0 { 3072 };
   int64_t encoder_layers_5_mlp_fc1_weight_dim_1 { 768 };
   int64_t encoder_layers_5_mlp_fc1_bias_dim_0 { 3072 };
   int64_t encoder_layers_5_mlp_fc2_weight_dim_0 { 768 };
   int64_t encoder_layers_5_mlp_fc2_weight_dim_1 { 3072 };
   int64_t encoder_layers_5_mlp_fc2_bias_dim_0 { 768 };
   int64_t reshape_90_0_dim_0 { 1 };
   int64_t reshape_90_0_dim_1 { 64 };
   int64_t reshape_90_0_dim_2 { 768 };
   int64_t encoder_layers_6_layer_norm1_weight_dim_0 { 768 };
   int64_t encoder_layers_6_layer_norm1_bias_dim_0 { 768 };
   int64_t encoder_layers_6_self_attn_qkv_weight_dim_0 { 2304 };
   int64_t encoder_layers_6_self_attn_qkv_weight_dim_1 { 768 };
   int64_t encoder_layers_6_self_attn_qkv_bias_dim_0 { 2304 };
   int64_t reshape_93_0_dim_0 { 64 };
   int64_t reshape_93_0_dim_1 { 3 };
   int64_t reshape_93_0_dim_2 { 12 };
   int64_t reshape_93_0_dim_3 { 64 };
   int64_t encoder_layers_6_self_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_94_0_dim_0 { 64 };
   int64_t flash_attention_94_0_dim_1 { 12 };
   int64_t flash_attention_94_0_dim_2 { 64 };
   int64_t encoder_layers_6_self_attn_proj_weight_dim_0 { 768 };
   int64_t encoder_layers_6_self_attn_proj_weight_dim_1 { 768 };
   int64_t encoder_layers_6_self_attn_proj_bias_dim_0 { 768 };
   int64_t reshape_97_0_dim_0 { 1 };
   int64_t reshape_97_0_dim_1 { 64 };
   int64_t reshape_97_0_dim_2 { 768 };
   int64_t reshape_95_0_dim_0 { 64 };
   int64_t reshape_95_0_dim_1 { 768 };
   int64_t encoder_layers_6_layer_norm2_weight_dim_0 { 768 };
   int64_t encoder_layers_6_layer_norm2_bias_dim_0 { 768 };
   int64_t encoder_layers_6_mlp_fc1_weight_dim_0 { 3072 };
   int64_t encoder_layers_6_mlp_fc1_weight_dim_1 { 768 };
   int64_t encoder_layers_6_mlp_fc1_bias_dim_0 { 3072 };
   int64_t encoder_layers_6_mlp_fc2_weight_dim_0 { 768 };
   int64_t encoder_layers_6_mlp_fc2_weight_dim_1 { 3072 };
   int64_t encoder_layers_6_mlp_fc2_bias_dim_0 { 768 };
   int64_t reshape_104_0_dim_0 { 1 };
   int64_t reshape_104_0_dim_1 { 64 };
   int64_t reshape_104_0_dim_2 { 768 };
   int64_t encoder_layers_7_layer_norm1_weight_dim_0 { 768 };
   int64_t encoder_layers_7_layer_norm1_bias_dim_0 { 768 };
   int64_t encoder_layers_7_self_attn_qkv_weight_dim_0 { 2304 };
   int64_t encoder_layers_7_self_attn_qkv_weight_dim_1 { 768 };
   int64_t encoder_layers_7_self_attn_qkv_bias_dim_0 { 2304 };
   int64_t reshape_107_0_dim_0 { 64 };
   int64_t reshape_107_0_dim_1 { 3 };
   int64_t reshape_107_0_dim_2 { 12 };
   int64_t reshape_107_0_dim_3 { 64 };
   int64_t encoder_layers_7_self_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_108_0_dim_0 { 64 };
   int64_t flash_attention_108_0_dim_1 { 12 };
   int64_t flash_attention_108_0_dim_2 { 64 };
   int64_t encoder_layers_7_self_attn_proj_weight_dim_0 { 768 };
   int64_t encoder_layers_7_self_attn_proj_weight_dim_1 { 768 };
   int64_t encoder_layers_7_self_attn_proj_bias_dim_0 { 768 };
   int64_t reshape_111_0_dim_0 { 1 };
   int64_t reshape_111_0_dim_1 { 64 };
   int64_t reshape_111_0_dim_2 { 768 };
   int64_t reshape_109_0_dim_0 { 64 };
   int64_t reshape_109_0_dim_1 { 768 };
   int64_t encoder_layers_7_layer_norm2_weight_dim_0 { 768 };
   int64_t encoder_layers_7_layer_norm2_bias_dim_0 { 768 };
   int64_t encoder_layers_7_mlp_fc1_weight_dim_0 { 3072 };
   int64_t encoder_layers_7_mlp_fc1_weight_dim_1 { 768 };
   int64_t encoder_layers_7_mlp_fc1_bias_dim_0 { 3072 };
   int64_t encoder_layers_7_mlp_fc2_weight_dim_0 { 768 };
   int64_t encoder_layers_7_mlp_fc2_weight_dim_1 { 3072 };
   int64_t encoder_layers_7_mlp_fc2_bias_dim_0 { 768 };
   int64_t reshape_118_0_dim_0 { 1 };
   int64_t reshape_118_0_dim_1 { 64 };
   int64_t reshape_118_0_dim_2 { 768 };
   int64_t encoder_layers_8_layer_norm1_weight_dim_0 { 768 };
   int64_t encoder_layers_8_layer_norm1_bias_dim_0 { 768 };
   int64_t encoder_layers_8_self_attn_qkv_weight_dim_0 { 2304 };
   int64_t encoder_layers_8_self_attn_qkv_weight_dim_1 { 768 };
   int64_t encoder_layers_8_self_attn_qkv_bias_dim_0 { 2304 };
   int64_t reshape_121_0_dim_0 { 64 };
   int64_t reshape_121_0_dim_1 { 3 };
   int64_t reshape_121_0_dim_2 { 12 };
   int64_t reshape_121_0_dim_3 { 64 };
   int64_t encoder_layers_8_self_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_122_0_dim_0 { 64 };
   int64_t flash_attention_122_0_dim_1 { 12 };
   int64_t flash_attention_122_0_dim_2 { 64 };
   int64_t encoder_layers_8_self_attn_proj_weight_dim_0 { 768 };
   int64_t encoder_layers_8_self_attn_proj_weight_dim_1 { 768 };
   int64_t encoder_layers_8_self_attn_proj_bias_dim_0 { 768 };
   int64_t reshape_125_0_dim_0 { 1 };
   int64_t reshape_125_0_dim_1 { 64 };
   int64_t reshape_125_0_dim_2 { 768 };
   int64_t reshape_123_0_dim_0 { 64 };
   int64_t reshape_123_0_dim_1 { 768 };
   int64_t encoder_layers_8_layer_norm2_weight_dim_0 { 768 };
   int64_t encoder_layers_8_layer_norm2_bias_dim_0 { 768 };
   int64_t encoder_layers_8_mlp_fc1_weight_dim_0 { 3072 };
   int64_t encoder_layers_8_mlp_fc1_weight_dim_1 { 768 };
   int64_t encoder_layers_8_mlp_fc1_bias_dim_0 { 3072 };
   int64_t encoder_layers_8_mlp_fc2_weight_dim_0 { 768 };
   int64_t encoder_layers_8_mlp_fc2_weight_dim_1 { 3072 };
   int64_t encoder_layers_8_mlp_fc2_bias_dim_0 { 768 };
   int64_t reshape_132_0_dim_0 { 1 };
   int64_t reshape_132_0_dim_1 { 64 };
   int64_t reshape_132_0_dim_2 { 768 };
   int64_t encoder_layers_9_layer_norm1_weight_dim_0 { 768 };
   int64_t encoder_layers_9_layer_norm1_bias_dim_0 { 768 };
   int64_t encoder_layers_9_self_attn_qkv_weight_dim_0 { 2304 };
   int64_t encoder_layers_9_self_attn_qkv_weight_dim_1 { 768 };
   int64_t encoder_layers_9_self_attn_qkv_bias_dim_0 { 2304 };
   int64_t reshape_135_0_dim_0 { 64 };
   int64_t reshape_135_0_dim_1 { 3 };
   int64_t reshape_135_0_dim_2 { 12 };
   int64_t reshape_135_0_dim_3 { 64 };
   int64_t encoder_layers_9_self_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_136_0_dim_0 { 64 };
   int64_t flash_attention_136_0_dim_1 { 12 };
   int64_t flash_attention_136_0_dim_2 { 64 };
   int64_t encoder_layers_9_self_attn_proj_weight_dim_0 { 768 };
   int64_t encoder_layers_9_self_attn_proj_weight_dim_1 { 768 };
   int64_t encoder_layers_9_self_attn_proj_bias_dim_0 { 768 };
   int64_t reshape_139_0_dim_0 { 1 };
   int64_t reshape_139_0_dim_1 { 64 };
   int64_t reshape_139_0_dim_2 { 768 };
   int64_t reshape_137_0_dim_0 { 64 };
   int64_t reshape_137_0_dim_1 { 768 };
   int64_t encoder_layers_9_layer_norm2_weight_dim_0 { 768 };
   int64_t encoder_layers_9_layer_norm2_bias_dim_0 { 768 };
   int64_t encoder_layers_9_mlp_fc1_weight_dim_0 { 3072 };
   int64_t encoder_layers_9_mlp_fc1_weight_dim_1 { 768 };
   int64_t encoder_layers_9_mlp_fc1_bias_dim_0 { 3072 };
   int64_t encoder_layers_9_mlp_fc2_weight_dim_0 { 768 };
   int64_t encoder_layers_9_mlp_fc2_weight_dim_1 { 3072 };
   int64_t encoder_layers_9_mlp_fc2_bias_dim_0 { 768 };
   int64_t reshape_146_0_dim_0 { 1 };
   int64_t reshape_146_0_dim_1 { 64 };
   int64_t reshape_146_0_dim_2 { 768 };
   int64_t encoder_layers_10_layer_norm1_weight_dim_0 { 768 };
   int64_t encoder_layers_10_layer_norm1_bias_dim_0 { 768 };
   int64_t encoder_layers_10_self_attn_qkv_weight_dim_0 { 2304 };
   int64_t encoder_layers_10_self_attn_qkv_weight_dim_1 { 768 };
   int64_t encoder_layers_10_self_attn_qkv_bias_dim_0 { 2304 };
   int64_t reshape_149_0_dim_0 { 64 };
   int64_t reshape_149_0_dim_1 { 3 };
   int64_t reshape_149_0_dim_2 { 12 };
   int64_t reshape_149_0_dim_3 { 64 };
   int64_t encoder_layers_10_self_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_150_0_dim_0 { 64 };
   int64_t flash_attention_150_0_dim_1 { 12 };
   int64_t flash_attention_150_0_dim_2 { 64 };
   int64_t encoder_layers_10_self_attn_proj_weight_dim_0 { 768 };
   int64_t encoder_layers_10_self_attn_proj_weight_dim_1 { 768 };
   int64_t encoder_layers_10_self_attn_proj_bias_dim_0 { 768 };
   int64_t reshape_153_0_dim_0 { 1 };
   int64_t reshape_153_0_dim_1 { 64 };
   int64_t reshape_153_0_dim_2 { 768 };
   int64_t reshape_151_0_dim_0 { 64 };
   int64_t reshape_151_0_dim_1 { 768 };
   int64_t encoder_layers_10_layer_norm2_weight_dim_0 { 768 };
   int64_t encoder_layers_10_layer_norm2_bias_dim_0 { 768 };
   int64_t encoder_layers_10_mlp_fc1_weight_dim_0 { 3072 };
   int64_t encoder_layers_10_mlp_fc1_weight_dim_1 { 768 };
   int64_t encoder_layers_10_mlp_fc1_bias_dim_0 { 3072 };
   int64_t encoder_layers_10_mlp_fc2_weight_dim_0 { 768 };
   int64_t encoder_layers_10_mlp_fc2_weight_dim_1 { 3072 };
   int64_t encoder_layers_10_mlp_fc2_bias_dim_0 { 768 };
   int64_t reshape_160_0_dim_0 { 1 };
   int64_t reshape_160_0_dim_1 { 64 };
   int64_t reshape_160_0_dim_2 { 768 };
   int64_t encoder_layers_11_layer_norm1_weight_dim_0 { 768 };
   int64_t encoder_layers_11_layer_norm1_bias_dim_0 { 768 };
   int64_t encoder_layers_11_self_attn_qkv_weight_dim_0 { 2304 };
   int64_t encoder_layers_11_self_attn_qkv_weight_dim_1 { 768 };
   int64_t encoder_layers_11_self_attn_qkv_bias_dim_0 { 2304 };
   int64_t reshape_163_0_dim_0 { 64 };
   int64_t reshape_163_0_dim_1 { 3 };
   int64_t reshape_163_0_dim_2 { 12 };
   int64_t reshape_163_0_dim_3 { 64 };
   int64_t encoder_layers_11_self_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_164_0_dim_0 { 64 };
   int64_t flash_attention_164_0_dim_1 { 12 };
   int64_t flash_attention_164_0_dim_2 { 64 };
   int64_t encoder_layers_11_self_attn_proj_weight_dim_0 { 768 };
   int64_t encoder_layers_11_self_attn_proj_weight_dim_1 { 768 };
   int64_t encoder_layers_11_self_attn_proj_bias_dim_0 { 768 };
   int64_t reshape_167_0_dim_0 { 1 };
   int64_t reshape_167_0_dim_1 { 64 };
   int64_t reshape_167_0_dim_2 { 768 };
   int64_t reshape_165_0_dim_0 { 64 };
   int64_t reshape_165_0_dim_1 { 768 };
   int64_t encoder_layers_11_layer_norm2_weight_dim_0 { 768 };
   int64_t encoder_layers_11_layer_norm2_bias_dim_0 { 768 };
   int64_t encoder_layers_11_mlp_fc1_weight_dim_0 { 3072 };
   int64_t encoder_layers_11_mlp_fc1_weight_dim_1 { 768 };
   int64_t encoder_layers_11_mlp_fc1_bias_dim_0 { 3072 };
   int64_t encoder_layers_11_mlp_fc2_weight_dim_0 { 768 };
   int64_t encoder_layers_11_mlp_fc2_weight_dim_1 { 3072 };
   int64_t encoder_layers_11_mlp_fc2_bias_dim_0 { 768 };
   int64_t reshape_174_0_dim_0 { 1 };
   int64_t reshape_174_0_dim_1 { 64 };
   int64_t reshape_174_0_dim_2 { 768 };
   int64_t final_layer_norm_weight_dim_0 { 768 };
   int64_t final_layer_norm_bias_dim_0 { 768 };

};
} // namespace ait