
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
    


void invoke_fused_elementwise_176(half* output0, const half* input0,const half* input1,  int n_elements, cudaStream_t stream);
    

void permute102_5(
  cutlass::half_t*,
  cutlass::half_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  cudaStream_t
);

    cudaError_t layernorm_6(half* output,
                   half* input,
                   const half* gamma,
                   const half* beta,
                   int m,
                   int n,
                   const float eps,
                   cudaStream_t stream);
    

void invoke_fused_elementwise_177(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void gemm_rcr_8(
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

void invoke_fused_elementwise_178(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void gemm_rcr_bias_19(
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

void invoke_fused_elementwise_180(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_182(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_184(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_186(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_188(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_190(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_192(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_194(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_196(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_198(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void invoke_fused_elementwise_200(half* output0, const half* input0,  int n_elements, cudaStream_t stream);
    

void permute102_174(
  cutlass::half_t*,
  cutlass::half_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  cudaStream_t
);

    cudaError_t layernorm_175(half* output,
                   half* input,
                   const half* gamma,
                   const half* beta,
                   int m,
                   int n,
                   const float eps,
                   cudaStream_t stream);
    

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

      token_embedding_weight = reinterpret_cast<decltype(token_embedding_weight)>(constants + 0);
    positional_embedding = reinterpret_cast<decltype(positional_embedding)>(constants + 50593792);
     constant_name_to_ptr_["transformer_resblocks_0_ln_1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_0_ln_1_weight));
     constant_name_to_ptr_["transformer_resblocks_0_ln_1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_0_ln_1_bias));
     constant_name_to_ptr_["transformer_resblocks_0_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_0_attn_qkv_weight));
     constant_name_to_ptr_["transformer_resblocks_0_attn_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_0_attn_cu_length));
     constant_name_to_ptr_["transformer_resblocks_0_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_0_attn_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_0_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_0_attn_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_0_ln_2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_0_ln_2_weight));
     constant_name_to_ptr_["transformer_resblocks_0_ln_2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_0_ln_2_bias));
     constant_name_to_ptr_["transformer_resblocks_0_mlp_c_fc_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_0_mlp_c_fc_weight));
     constant_name_to_ptr_["transformer_resblocks_0_mlp_c_fc_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_0_mlp_c_fc_bias));
     constant_name_to_ptr_["transformer_resblocks_0_mlp_c_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_0_mlp_c_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_0_mlp_c_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_0_mlp_c_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_1_ln_1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_1_ln_1_weight));
     constant_name_to_ptr_["transformer_resblocks_1_ln_1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_1_ln_1_bias));
     constant_name_to_ptr_["transformer_resblocks_1_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_1_attn_qkv_weight));
     constant_name_to_ptr_["transformer_resblocks_1_attn_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_1_attn_cu_length));
     constant_name_to_ptr_["transformer_resblocks_1_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_1_attn_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_1_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_1_attn_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_1_ln_2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_1_ln_2_weight));
     constant_name_to_ptr_["transformer_resblocks_1_ln_2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_1_ln_2_bias));
     constant_name_to_ptr_["transformer_resblocks_1_mlp_c_fc_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_1_mlp_c_fc_weight));
     constant_name_to_ptr_["transformer_resblocks_1_mlp_c_fc_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_1_mlp_c_fc_bias));
     constant_name_to_ptr_["transformer_resblocks_1_mlp_c_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_1_mlp_c_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_1_mlp_c_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_1_mlp_c_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_2_ln_1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_2_ln_1_weight));
     constant_name_to_ptr_["transformer_resblocks_2_ln_1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_2_ln_1_bias));
     constant_name_to_ptr_["transformer_resblocks_2_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_2_attn_qkv_weight));
     constant_name_to_ptr_["transformer_resblocks_2_attn_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_2_attn_cu_length));
     constant_name_to_ptr_["transformer_resblocks_2_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_2_attn_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_2_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_2_attn_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_2_ln_2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_2_ln_2_weight));
     constant_name_to_ptr_["transformer_resblocks_2_ln_2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_2_ln_2_bias));
     constant_name_to_ptr_["transformer_resblocks_2_mlp_c_fc_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_2_mlp_c_fc_weight));
     constant_name_to_ptr_["transformer_resblocks_2_mlp_c_fc_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_2_mlp_c_fc_bias));
     constant_name_to_ptr_["transformer_resblocks_2_mlp_c_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_2_mlp_c_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_2_mlp_c_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_2_mlp_c_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_3_ln_1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_3_ln_1_weight));
     constant_name_to_ptr_["transformer_resblocks_3_ln_1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_3_ln_1_bias));
     constant_name_to_ptr_["transformer_resblocks_3_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_3_attn_qkv_weight));
     constant_name_to_ptr_["transformer_resblocks_3_attn_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_3_attn_cu_length));
     constant_name_to_ptr_["transformer_resblocks_3_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_3_attn_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_3_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_3_attn_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_3_ln_2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_3_ln_2_weight));
     constant_name_to_ptr_["transformer_resblocks_3_ln_2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_3_ln_2_bias));
     constant_name_to_ptr_["transformer_resblocks_3_mlp_c_fc_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_3_mlp_c_fc_weight));
     constant_name_to_ptr_["transformer_resblocks_3_mlp_c_fc_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_3_mlp_c_fc_bias));
     constant_name_to_ptr_["transformer_resblocks_3_mlp_c_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_3_mlp_c_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_3_mlp_c_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_3_mlp_c_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_4_ln_1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_4_ln_1_weight));
     constant_name_to_ptr_["transformer_resblocks_4_ln_1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_4_ln_1_bias));
     constant_name_to_ptr_["transformer_resblocks_4_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_4_attn_qkv_weight));
     constant_name_to_ptr_["transformer_resblocks_4_attn_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_4_attn_cu_length));
     constant_name_to_ptr_["transformer_resblocks_4_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_4_attn_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_4_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_4_attn_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_4_ln_2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_4_ln_2_weight));
     constant_name_to_ptr_["transformer_resblocks_4_ln_2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_4_ln_2_bias));
     constant_name_to_ptr_["transformer_resblocks_4_mlp_c_fc_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_4_mlp_c_fc_weight));
     constant_name_to_ptr_["transformer_resblocks_4_mlp_c_fc_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_4_mlp_c_fc_bias));
     constant_name_to_ptr_["transformer_resblocks_4_mlp_c_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_4_mlp_c_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_4_mlp_c_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_4_mlp_c_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_5_ln_1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_5_ln_1_weight));
     constant_name_to_ptr_["transformer_resblocks_5_ln_1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_5_ln_1_bias));
     constant_name_to_ptr_["transformer_resblocks_5_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_5_attn_qkv_weight));
     constant_name_to_ptr_["transformer_resblocks_5_attn_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_5_attn_cu_length));
     constant_name_to_ptr_["transformer_resblocks_5_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_5_attn_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_5_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_5_attn_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_5_ln_2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_5_ln_2_weight));
     constant_name_to_ptr_["transformer_resblocks_5_ln_2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_5_ln_2_bias));
     constant_name_to_ptr_["transformer_resblocks_5_mlp_c_fc_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_5_mlp_c_fc_weight));
     constant_name_to_ptr_["transformer_resblocks_5_mlp_c_fc_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_5_mlp_c_fc_bias));
     constant_name_to_ptr_["transformer_resblocks_5_mlp_c_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_5_mlp_c_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_5_mlp_c_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_5_mlp_c_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_6_ln_1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_6_ln_1_weight));
     constant_name_to_ptr_["transformer_resblocks_6_ln_1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_6_ln_1_bias));
     constant_name_to_ptr_["transformer_resblocks_6_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_6_attn_qkv_weight));
     constant_name_to_ptr_["transformer_resblocks_6_attn_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_6_attn_cu_length));
     constant_name_to_ptr_["transformer_resblocks_6_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_6_attn_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_6_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_6_attn_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_6_ln_2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_6_ln_2_weight));
     constant_name_to_ptr_["transformer_resblocks_6_ln_2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_6_ln_2_bias));
     constant_name_to_ptr_["transformer_resblocks_6_mlp_c_fc_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_6_mlp_c_fc_weight));
     constant_name_to_ptr_["transformer_resblocks_6_mlp_c_fc_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_6_mlp_c_fc_bias));
     constant_name_to_ptr_["transformer_resblocks_6_mlp_c_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_6_mlp_c_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_6_mlp_c_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_6_mlp_c_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_7_ln_1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_7_ln_1_weight));
     constant_name_to_ptr_["transformer_resblocks_7_ln_1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_7_ln_1_bias));
     constant_name_to_ptr_["transformer_resblocks_7_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_7_attn_qkv_weight));
     constant_name_to_ptr_["transformer_resblocks_7_attn_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_7_attn_cu_length));
     constant_name_to_ptr_["transformer_resblocks_7_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_7_attn_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_7_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_7_attn_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_7_ln_2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_7_ln_2_weight));
     constant_name_to_ptr_["transformer_resblocks_7_ln_2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_7_ln_2_bias));
     constant_name_to_ptr_["transformer_resblocks_7_mlp_c_fc_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_7_mlp_c_fc_weight));
     constant_name_to_ptr_["transformer_resblocks_7_mlp_c_fc_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_7_mlp_c_fc_bias));
     constant_name_to_ptr_["transformer_resblocks_7_mlp_c_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_7_mlp_c_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_7_mlp_c_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_7_mlp_c_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_8_ln_1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_8_ln_1_weight));
     constant_name_to_ptr_["transformer_resblocks_8_ln_1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_8_ln_1_bias));
     constant_name_to_ptr_["transformer_resblocks_8_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_8_attn_qkv_weight));
     constant_name_to_ptr_["transformer_resblocks_8_attn_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_8_attn_cu_length));
     constant_name_to_ptr_["transformer_resblocks_8_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_8_attn_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_8_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_8_attn_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_8_ln_2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_8_ln_2_weight));
     constant_name_to_ptr_["transformer_resblocks_8_ln_2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_8_ln_2_bias));
     constant_name_to_ptr_["transformer_resblocks_8_mlp_c_fc_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_8_mlp_c_fc_weight));
     constant_name_to_ptr_["transformer_resblocks_8_mlp_c_fc_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_8_mlp_c_fc_bias));
     constant_name_to_ptr_["transformer_resblocks_8_mlp_c_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_8_mlp_c_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_8_mlp_c_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_8_mlp_c_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_9_ln_1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_9_ln_1_weight));
     constant_name_to_ptr_["transformer_resblocks_9_ln_1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_9_ln_1_bias));
     constant_name_to_ptr_["transformer_resblocks_9_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_9_attn_qkv_weight));
     constant_name_to_ptr_["transformer_resblocks_9_attn_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_9_attn_cu_length));
     constant_name_to_ptr_["transformer_resblocks_9_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_9_attn_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_9_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_9_attn_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_9_ln_2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_9_ln_2_weight));
     constant_name_to_ptr_["transformer_resblocks_9_ln_2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_9_ln_2_bias));
     constant_name_to_ptr_["transformer_resblocks_9_mlp_c_fc_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_9_mlp_c_fc_weight));
     constant_name_to_ptr_["transformer_resblocks_9_mlp_c_fc_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_9_mlp_c_fc_bias));
     constant_name_to_ptr_["transformer_resblocks_9_mlp_c_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_9_mlp_c_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_9_mlp_c_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_9_mlp_c_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_10_ln_1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_10_ln_1_weight));
     constant_name_to_ptr_["transformer_resblocks_10_ln_1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_10_ln_1_bias));
     constant_name_to_ptr_["transformer_resblocks_10_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_10_attn_qkv_weight));
     constant_name_to_ptr_["transformer_resblocks_10_attn_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_10_attn_cu_length));
     constant_name_to_ptr_["transformer_resblocks_10_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_10_attn_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_10_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_10_attn_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_10_ln_2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_10_ln_2_weight));
     constant_name_to_ptr_["transformer_resblocks_10_ln_2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_10_ln_2_bias));
     constant_name_to_ptr_["transformer_resblocks_10_mlp_c_fc_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_10_mlp_c_fc_weight));
     constant_name_to_ptr_["transformer_resblocks_10_mlp_c_fc_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_10_mlp_c_fc_bias));
     constant_name_to_ptr_["transformer_resblocks_10_mlp_c_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_10_mlp_c_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_10_mlp_c_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_10_mlp_c_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_11_ln_1_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_11_ln_1_weight));
     constant_name_to_ptr_["transformer_resblocks_11_ln_1_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_11_ln_1_bias));
     constant_name_to_ptr_["transformer_resblocks_11_attn_qkv_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_11_attn_qkv_weight));
     constant_name_to_ptr_["transformer_resblocks_11_attn_cu_length"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_11_attn_cu_length));
     constant_name_to_ptr_["transformer_resblocks_11_attn_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_11_attn_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_11_attn_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_11_attn_proj_bias));
     constant_name_to_ptr_["transformer_resblocks_11_ln_2_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_11_ln_2_weight));
     constant_name_to_ptr_["transformer_resblocks_11_ln_2_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_11_ln_2_bias));
     constant_name_to_ptr_["transformer_resblocks_11_mlp_c_fc_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_11_mlp_c_fc_weight));
     constant_name_to_ptr_["transformer_resblocks_11_mlp_c_fc_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_11_mlp_c_fc_bias));
     constant_name_to_ptr_["transformer_resblocks_11_mlp_c_proj_weight"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_11_mlp_c_proj_weight));
     constant_name_to_ptr_["transformer_resblocks_11_mlp_c_proj_bias"] = const_cast<const void**>(reinterpret_cast<void**>(&transformer_resblocks_11_mlp_c_proj_bias));
    ln_final_weight = reinterpret_cast<decltype(ln_final_weight)>(constants + 50672640);
    ln_final_bias = reinterpret_cast<decltype(ln_final_bias)>(constants + 50673664);
      auto* blob_ptr = static_cast<uint8_t*>(blob.get());
      batch_gather_1_0 = reinterpret_cast<decltype(batch_gather_1_0)>(blob_ptr + 0);
    size_2_0 = reinterpret_cast<decltype(size_2_0)>(blob_ptr + 79488);
    size_2_1 = reinterpret_cast<decltype(size_2_1)>(blob_ptr + 79552);
    elementwise_4_0 = reinterpret_cast<decltype(elementwise_4_0)>(blob_ptr + 78848);
    permute102_5_0 = reinterpret_cast<decltype(permute102_5_0)>(blob_ptr + 236544);
    layernorm_6_0 = reinterpret_cast<decltype(layernorm_6_0)>(blob_ptr + 0);
    elementwise_7_0 = reinterpret_cast<decltype(elementwise_7_0)>(blob_ptr + 315392);
    reshape_9_0 = reinterpret_cast<decltype(reshape_9_0)>(blob_ptr + 0);
    flash_attention_10_0 = reinterpret_cast<decltype(flash_attention_10_0)>(blob_ptr + 315392);
    reshape_13_0 = reinterpret_cast<decltype(reshape_13_0)>(blob_ptr + 0);
    layernorm_14_0 = reinterpret_cast<decltype(layernorm_14_0)>(blob_ptr + 315392);
    gemm_rcr_bias_15_0 = reinterpret_cast<decltype(gemm_rcr_bias_15_0)>(blob_ptr + 0);
    elementwise_18_0 = reinterpret_cast<decltype(elementwise_18_0)>(blob_ptr + 315392);
    gemm_rcr_bias_19_0 = reinterpret_cast<decltype(gemm_rcr_bias_19_0)>(blob_ptr + 236544);
    layernorm_20_0 = reinterpret_cast<decltype(layernorm_20_0)>(blob_ptr + 0);
    elementwise_21_0 = reinterpret_cast<decltype(elementwise_21_0)>(blob_ptr + 315392);
    reshape_23_0 = reinterpret_cast<decltype(reshape_23_0)>(blob_ptr + 0);
    flash_attention_24_0 = reinterpret_cast<decltype(flash_attention_24_0)>(blob_ptr + 315392);
    reshape_27_0 = reinterpret_cast<decltype(reshape_27_0)>(blob_ptr + 0);
    layernorm_28_0 = reinterpret_cast<decltype(layernorm_28_0)>(blob_ptr + 315392);
    gemm_rcr_bias_29_0 = reinterpret_cast<decltype(gemm_rcr_bias_29_0)>(blob_ptr + 0);
    elementwise_32_0 = reinterpret_cast<decltype(elementwise_32_0)>(blob_ptr + 315392);
    gemm_rcr_bias_33_0 = reinterpret_cast<decltype(gemm_rcr_bias_33_0)>(blob_ptr + 236544);
    layernorm_34_0 = reinterpret_cast<decltype(layernorm_34_0)>(blob_ptr + 0);
    elementwise_35_0 = reinterpret_cast<decltype(elementwise_35_0)>(blob_ptr + 315392);
    reshape_37_0 = reinterpret_cast<decltype(reshape_37_0)>(blob_ptr + 0);
    flash_attention_38_0 = reinterpret_cast<decltype(flash_attention_38_0)>(blob_ptr + 315392);
    reshape_41_0 = reinterpret_cast<decltype(reshape_41_0)>(blob_ptr + 0);
    layernorm_42_0 = reinterpret_cast<decltype(layernorm_42_0)>(blob_ptr + 315392);
    gemm_rcr_bias_43_0 = reinterpret_cast<decltype(gemm_rcr_bias_43_0)>(blob_ptr + 0);
    elementwise_46_0 = reinterpret_cast<decltype(elementwise_46_0)>(blob_ptr + 315392);
    gemm_rcr_bias_47_0 = reinterpret_cast<decltype(gemm_rcr_bias_47_0)>(blob_ptr + 236544);
    layernorm_48_0 = reinterpret_cast<decltype(layernorm_48_0)>(blob_ptr + 0);
    elementwise_49_0 = reinterpret_cast<decltype(elementwise_49_0)>(blob_ptr + 315392);
    reshape_51_0 = reinterpret_cast<decltype(reshape_51_0)>(blob_ptr + 0);
    flash_attention_52_0 = reinterpret_cast<decltype(flash_attention_52_0)>(blob_ptr + 315392);
    reshape_55_0 = reinterpret_cast<decltype(reshape_55_0)>(blob_ptr + 0);
    layernorm_56_0 = reinterpret_cast<decltype(layernorm_56_0)>(blob_ptr + 315392);
    gemm_rcr_bias_57_0 = reinterpret_cast<decltype(gemm_rcr_bias_57_0)>(blob_ptr + 0);
    elementwise_60_0 = reinterpret_cast<decltype(elementwise_60_0)>(blob_ptr + 315392);
    gemm_rcr_bias_61_0 = reinterpret_cast<decltype(gemm_rcr_bias_61_0)>(blob_ptr + 236544);
    layernorm_62_0 = reinterpret_cast<decltype(layernorm_62_0)>(blob_ptr + 0);
    elementwise_63_0 = reinterpret_cast<decltype(elementwise_63_0)>(blob_ptr + 315392);
    reshape_65_0 = reinterpret_cast<decltype(reshape_65_0)>(blob_ptr + 0);
    flash_attention_66_0 = reinterpret_cast<decltype(flash_attention_66_0)>(blob_ptr + 315392);
    reshape_69_0 = reinterpret_cast<decltype(reshape_69_0)>(blob_ptr + 0);
    layernorm_70_0 = reinterpret_cast<decltype(layernorm_70_0)>(blob_ptr + 315392);
    gemm_rcr_bias_71_0 = reinterpret_cast<decltype(gemm_rcr_bias_71_0)>(blob_ptr + 0);
    elementwise_74_0 = reinterpret_cast<decltype(elementwise_74_0)>(blob_ptr + 315392);
    gemm_rcr_bias_75_0 = reinterpret_cast<decltype(gemm_rcr_bias_75_0)>(blob_ptr + 236544);
    layernorm_76_0 = reinterpret_cast<decltype(layernorm_76_0)>(blob_ptr + 0);
    elementwise_77_0 = reinterpret_cast<decltype(elementwise_77_0)>(blob_ptr + 315392);
    reshape_79_0 = reinterpret_cast<decltype(reshape_79_0)>(blob_ptr + 0);
    flash_attention_80_0 = reinterpret_cast<decltype(flash_attention_80_0)>(blob_ptr + 315392);
    reshape_83_0 = reinterpret_cast<decltype(reshape_83_0)>(blob_ptr + 0);
    layernorm_84_0 = reinterpret_cast<decltype(layernorm_84_0)>(blob_ptr + 315392);
    gemm_rcr_bias_85_0 = reinterpret_cast<decltype(gemm_rcr_bias_85_0)>(blob_ptr + 0);
    elementwise_88_0 = reinterpret_cast<decltype(elementwise_88_0)>(blob_ptr + 315392);
    gemm_rcr_bias_89_0 = reinterpret_cast<decltype(gemm_rcr_bias_89_0)>(blob_ptr + 236544);
    layernorm_90_0 = reinterpret_cast<decltype(layernorm_90_0)>(blob_ptr + 0);
    elementwise_91_0 = reinterpret_cast<decltype(elementwise_91_0)>(blob_ptr + 315392);
    reshape_93_0 = reinterpret_cast<decltype(reshape_93_0)>(blob_ptr + 0);
    flash_attention_94_0 = reinterpret_cast<decltype(flash_attention_94_0)>(blob_ptr + 315392);
    reshape_97_0 = reinterpret_cast<decltype(reshape_97_0)>(blob_ptr + 0);
    layernorm_98_0 = reinterpret_cast<decltype(layernorm_98_0)>(blob_ptr + 315392);
    gemm_rcr_bias_99_0 = reinterpret_cast<decltype(gemm_rcr_bias_99_0)>(blob_ptr + 0);
    elementwise_102_0 = reinterpret_cast<decltype(elementwise_102_0)>(blob_ptr + 315392);
    gemm_rcr_bias_103_0 = reinterpret_cast<decltype(gemm_rcr_bias_103_0)>(blob_ptr + 236544);
    layernorm_104_0 = reinterpret_cast<decltype(layernorm_104_0)>(blob_ptr + 0);
    elementwise_105_0 = reinterpret_cast<decltype(elementwise_105_0)>(blob_ptr + 315392);
    reshape_107_0 = reinterpret_cast<decltype(reshape_107_0)>(blob_ptr + 0);
    flash_attention_108_0 = reinterpret_cast<decltype(flash_attention_108_0)>(blob_ptr + 315392);
    reshape_111_0 = reinterpret_cast<decltype(reshape_111_0)>(blob_ptr + 0);
    layernorm_112_0 = reinterpret_cast<decltype(layernorm_112_0)>(blob_ptr + 315392);
    gemm_rcr_bias_113_0 = reinterpret_cast<decltype(gemm_rcr_bias_113_0)>(blob_ptr + 0);
    elementwise_116_0 = reinterpret_cast<decltype(elementwise_116_0)>(blob_ptr + 315392);
    gemm_rcr_bias_117_0 = reinterpret_cast<decltype(gemm_rcr_bias_117_0)>(blob_ptr + 236544);
    layernorm_118_0 = reinterpret_cast<decltype(layernorm_118_0)>(blob_ptr + 0);
    elementwise_119_0 = reinterpret_cast<decltype(elementwise_119_0)>(blob_ptr + 315392);
    reshape_121_0 = reinterpret_cast<decltype(reshape_121_0)>(blob_ptr + 0);
    flash_attention_122_0 = reinterpret_cast<decltype(flash_attention_122_0)>(blob_ptr + 315392);
    reshape_125_0 = reinterpret_cast<decltype(reshape_125_0)>(blob_ptr + 0);
    layernorm_126_0 = reinterpret_cast<decltype(layernorm_126_0)>(blob_ptr + 315392);
    gemm_rcr_bias_127_0 = reinterpret_cast<decltype(gemm_rcr_bias_127_0)>(blob_ptr + 0);
    elementwise_130_0 = reinterpret_cast<decltype(elementwise_130_0)>(blob_ptr + 315392);
    gemm_rcr_bias_131_0 = reinterpret_cast<decltype(gemm_rcr_bias_131_0)>(blob_ptr + 236544);
    layernorm_132_0 = reinterpret_cast<decltype(layernorm_132_0)>(blob_ptr + 0);
    elementwise_133_0 = reinterpret_cast<decltype(elementwise_133_0)>(blob_ptr + 315392);
    reshape_135_0 = reinterpret_cast<decltype(reshape_135_0)>(blob_ptr + 0);
    flash_attention_136_0 = reinterpret_cast<decltype(flash_attention_136_0)>(blob_ptr + 315392);
    reshape_139_0 = reinterpret_cast<decltype(reshape_139_0)>(blob_ptr + 0);
    layernorm_140_0 = reinterpret_cast<decltype(layernorm_140_0)>(blob_ptr + 315392);
    gemm_rcr_bias_141_0 = reinterpret_cast<decltype(gemm_rcr_bias_141_0)>(blob_ptr + 0);
    elementwise_144_0 = reinterpret_cast<decltype(elementwise_144_0)>(blob_ptr + 315392);
    gemm_rcr_bias_145_0 = reinterpret_cast<decltype(gemm_rcr_bias_145_0)>(blob_ptr + 236544);
    layernorm_146_0 = reinterpret_cast<decltype(layernorm_146_0)>(blob_ptr + 0);
    elementwise_147_0 = reinterpret_cast<decltype(elementwise_147_0)>(blob_ptr + 315392);
    reshape_149_0 = reinterpret_cast<decltype(reshape_149_0)>(blob_ptr + 0);
    flash_attention_150_0 = reinterpret_cast<decltype(flash_attention_150_0)>(blob_ptr + 315392);
    reshape_153_0 = reinterpret_cast<decltype(reshape_153_0)>(blob_ptr + 0);
    layernorm_154_0 = reinterpret_cast<decltype(layernorm_154_0)>(blob_ptr + 315392);
    gemm_rcr_bias_155_0 = reinterpret_cast<decltype(gemm_rcr_bias_155_0)>(blob_ptr + 0);
    elementwise_158_0 = reinterpret_cast<decltype(elementwise_158_0)>(blob_ptr + 315392);
    gemm_rcr_bias_159_0 = reinterpret_cast<decltype(gemm_rcr_bias_159_0)>(blob_ptr + 236544);
    layernorm_160_0 = reinterpret_cast<decltype(layernorm_160_0)>(blob_ptr + 0);
    elementwise_161_0 = reinterpret_cast<decltype(elementwise_161_0)>(blob_ptr + 315392);
    reshape_163_0 = reinterpret_cast<decltype(reshape_163_0)>(blob_ptr + 0);
    flash_attention_164_0 = reinterpret_cast<decltype(flash_attention_164_0)>(blob_ptr + 315392);
    reshape_167_0 = reinterpret_cast<decltype(reshape_167_0)>(blob_ptr + 0);
    layernorm_168_0 = reinterpret_cast<decltype(layernorm_168_0)>(blob_ptr + 315392);
    gemm_rcr_bias_169_0 = reinterpret_cast<decltype(gemm_rcr_bias_169_0)>(blob_ptr + 0);
    elementwise_172_0 = reinterpret_cast<decltype(elementwise_172_0)>(blob_ptr + 315392);
    gemm_rcr_bias_173_0 = reinterpret_cast<decltype(gemm_rcr_bias_173_0)>(blob_ptr + 0);
    permute102_174_0 = reinterpret_cast<decltype(permute102_174_0)>(blob_ptr + 78848);
  
       params[0].shape_ptrs = {ParamDim(1, 1, &input_dim_0), ParamDim(77, 77, &input_dim_1)};
     params[2].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_0_ln_1_weight_dim_0)};
     params[3].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_0_ln_1_bias_dim_0)};
     params[4].shape_ptrs = {ParamDim(1536, 1536, &transformer_resblocks_0_attn_qkv_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_0_attn_qkv_weight_dim_1)};
     params[5].shape_ptrs = {ParamDim(2, 2, &transformer_resblocks_0_attn_cu_length_dim_0)};
     params[6].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_0_attn_proj_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_0_attn_proj_weight_dim_1)};
     params[7].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_0_attn_proj_bias_dim_0)};
     params[8].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_0_ln_2_weight_dim_0)};
     params[9].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_0_ln_2_bias_dim_0)};
     params[10].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_0_mlp_c_fc_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_0_mlp_c_fc_weight_dim_1)};
     params[11].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_0_mlp_c_fc_bias_dim_0)};
     params[12].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_0_mlp_c_proj_weight_dim_0), ParamDim(2048, 2048, &transformer_resblocks_0_mlp_c_proj_weight_dim_1)};
     params[13].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_0_mlp_c_proj_bias_dim_0)};
     params[14].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_1_ln_1_weight_dim_0)};
     params[15].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_1_ln_1_bias_dim_0)};
     params[16].shape_ptrs = {ParamDim(1536, 1536, &transformer_resblocks_1_attn_qkv_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_1_attn_qkv_weight_dim_1)};
     params[17].shape_ptrs = {ParamDim(2, 2, &transformer_resblocks_1_attn_cu_length_dim_0)};
     params[18].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_1_attn_proj_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_1_attn_proj_weight_dim_1)};
     params[19].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_1_attn_proj_bias_dim_0)};
     params[20].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_1_ln_2_weight_dim_0)};
     params[21].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_1_ln_2_bias_dim_0)};
     params[22].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_1_mlp_c_fc_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_1_mlp_c_fc_weight_dim_1)};
     params[23].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_1_mlp_c_fc_bias_dim_0)};
     params[24].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_1_mlp_c_proj_weight_dim_0), ParamDim(2048, 2048, &transformer_resblocks_1_mlp_c_proj_weight_dim_1)};
     params[25].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_1_mlp_c_proj_bias_dim_0)};
     params[26].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_2_ln_1_weight_dim_0)};
     params[27].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_2_ln_1_bias_dim_0)};
     params[28].shape_ptrs = {ParamDim(1536, 1536, &transformer_resblocks_2_attn_qkv_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_2_attn_qkv_weight_dim_1)};
     params[29].shape_ptrs = {ParamDim(2, 2, &transformer_resblocks_2_attn_cu_length_dim_0)};
     params[30].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_2_attn_proj_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_2_attn_proj_weight_dim_1)};
     params[31].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_2_attn_proj_bias_dim_0)};
     params[32].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_2_ln_2_weight_dim_0)};
     params[33].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_2_ln_2_bias_dim_0)};
     params[34].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_2_mlp_c_fc_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_2_mlp_c_fc_weight_dim_1)};
     params[35].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_2_mlp_c_fc_bias_dim_0)};
     params[36].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_2_mlp_c_proj_weight_dim_0), ParamDim(2048, 2048, &transformer_resblocks_2_mlp_c_proj_weight_dim_1)};
     params[37].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_2_mlp_c_proj_bias_dim_0)};
     params[38].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_3_ln_1_weight_dim_0)};
     params[39].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_3_ln_1_bias_dim_0)};
     params[40].shape_ptrs = {ParamDim(1536, 1536, &transformer_resblocks_3_attn_qkv_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_3_attn_qkv_weight_dim_1)};
     params[41].shape_ptrs = {ParamDim(2, 2, &transformer_resblocks_3_attn_cu_length_dim_0)};
     params[42].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_3_attn_proj_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_3_attn_proj_weight_dim_1)};
     params[43].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_3_attn_proj_bias_dim_0)};
     params[44].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_3_ln_2_weight_dim_0)};
     params[45].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_3_ln_2_bias_dim_0)};
     params[46].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_3_mlp_c_fc_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_3_mlp_c_fc_weight_dim_1)};
     params[47].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_3_mlp_c_fc_bias_dim_0)};
     params[48].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_3_mlp_c_proj_weight_dim_0), ParamDim(2048, 2048, &transformer_resblocks_3_mlp_c_proj_weight_dim_1)};
     params[49].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_3_mlp_c_proj_bias_dim_0)};
     params[50].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_4_ln_1_weight_dim_0)};
     params[51].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_4_ln_1_bias_dim_0)};
     params[52].shape_ptrs = {ParamDim(1536, 1536, &transformer_resblocks_4_attn_qkv_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_4_attn_qkv_weight_dim_1)};
     params[53].shape_ptrs = {ParamDim(2, 2, &transformer_resblocks_4_attn_cu_length_dim_0)};
     params[54].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_4_attn_proj_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_4_attn_proj_weight_dim_1)};
     params[55].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_4_attn_proj_bias_dim_0)};
     params[56].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_4_ln_2_weight_dim_0)};
     params[57].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_4_ln_2_bias_dim_0)};
     params[58].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_4_mlp_c_fc_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_4_mlp_c_fc_weight_dim_1)};
     params[59].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_4_mlp_c_fc_bias_dim_0)};
     params[60].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_4_mlp_c_proj_weight_dim_0), ParamDim(2048, 2048, &transformer_resblocks_4_mlp_c_proj_weight_dim_1)};
     params[61].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_4_mlp_c_proj_bias_dim_0)};
     params[62].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_5_ln_1_weight_dim_0)};
     params[63].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_5_ln_1_bias_dim_0)};
     params[64].shape_ptrs = {ParamDim(1536, 1536, &transformer_resblocks_5_attn_qkv_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_5_attn_qkv_weight_dim_1)};
     params[65].shape_ptrs = {ParamDim(2, 2, &transformer_resblocks_5_attn_cu_length_dim_0)};
     params[66].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_5_attn_proj_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_5_attn_proj_weight_dim_1)};
     params[67].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_5_attn_proj_bias_dim_0)};
     params[68].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_5_ln_2_weight_dim_0)};
     params[69].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_5_ln_2_bias_dim_0)};
     params[70].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_5_mlp_c_fc_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_5_mlp_c_fc_weight_dim_1)};
     params[71].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_5_mlp_c_fc_bias_dim_0)};
     params[72].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_5_mlp_c_proj_weight_dim_0), ParamDim(2048, 2048, &transformer_resblocks_5_mlp_c_proj_weight_dim_1)};
     params[73].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_5_mlp_c_proj_bias_dim_0)};
     params[74].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_6_ln_1_weight_dim_0)};
     params[75].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_6_ln_1_bias_dim_0)};
     params[76].shape_ptrs = {ParamDim(1536, 1536, &transformer_resblocks_6_attn_qkv_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_6_attn_qkv_weight_dim_1)};
     params[77].shape_ptrs = {ParamDim(2, 2, &transformer_resblocks_6_attn_cu_length_dim_0)};
     params[78].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_6_attn_proj_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_6_attn_proj_weight_dim_1)};
     params[79].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_6_attn_proj_bias_dim_0)};
     params[80].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_6_ln_2_weight_dim_0)};
     params[81].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_6_ln_2_bias_dim_0)};
     params[82].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_6_mlp_c_fc_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_6_mlp_c_fc_weight_dim_1)};
     params[83].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_6_mlp_c_fc_bias_dim_0)};
     params[84].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_6_mlp_c_proj_weight_dim_0), ParamDim(2048, 2048, &transformer_resblocks_6_mlp_c_proj_weight_dim_1)};
     params[85].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_6_mlp_c_proj_bias_dim_0)};
     params[86].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_7_ln_1_weight_dim_0)};
     params[87].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_7_ln_1_bias_dim_0)};
     params[88].shape_ptrs = {ParamDim(1536, 1536, &transformer_resblocks_7_attn_qkv_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_7_attn_qkv_weight_dim_1)};
     params[89].shape_ptrs = {ParamDim(2, 2, &transformer_resblocks_7_attn_cu_length_dim_0)};
     params[90].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_7_attn_proj_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_7_attn_proj_weight_dim_1)};
     params[91].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_7_attn_proj_bias_dim_0)};
     params[92].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_7_ln_2_weight_dim_0)};
     params[93].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_7_ln_2_bias_dim_0)};
     params[94].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_7_mlp_c_fc_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_7_mlp_c_fc_weight_dim_1)};
     params[95].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_7_mlp_c_fc_bias_dim_0)};
     params[96].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_7_mlp_c_proj_weight_dim_0), ParamDim(2048, 2048, &transformer_resblocks_7_mlp_c_proj_weight_dim_1)};
     params[97].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_7_mlp_c_proj_bias_dim_0)};
     params[98].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_8_ln_1_weight_dim_0)};
     params[99].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_8_ln_1_bias_dim_0)};
     params[100].shape_ptrs = {ParamDim(1536, 1536, &transformer_resblocks_8_attn_qkv_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_8_attn_qkv_weight_dim_1)};
     params[101].shape_ptrs = {ParamDim(2, 2, &transformer_resblocks_8_attn_cu_length_dim_0)};
     params[102].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_8_attn_proj_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_8_attn_proj_weight_dim_1)};
     params[103].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_8_attn_proj_bias_dim_0)};
     params[104].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_8_ln_2_weight_dim_0)};
     params[105].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_8_ln_2_bias_dim_0)};
     params[106].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_8_mlp_c_fc_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_8_mlp_c_fc_weight_dim_1)};
     params[107].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_8_mlp_c_fc_bias_dim_0)};
     params[108].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_8_mlp_c_proj_weight_dim_0), ParamDim(2048, 2048, &transformer_resblocks_8_mlp_c_proj_weight_dim_1)};
     params[109].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_8_mlp_c_proj_bias_dim_0)};
     params[110].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_9_ln_1_weight_dim_0)};
     params[111].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_9_ln_1_bias_dim_0)};
     params[112].shape_ptrs = {ParamDim(1536, 1536, &transformer_resblocks_9_attn_qkv_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_9_attn_qkv_weight_dim_1)};
     params[113].shape_ptrs = {ParamDim(2, 2, &transformer_resblocks_9_attn_cu_length_dim_0)};
     params[114].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_9_attn_proj_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_9_attn_proj_weight_dim_1)};
     params[115].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_9_attn_proj_bias_dim_0)};
     params[116].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_9_ln_2_weight_dim_0)};
     params[117].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_9_ln_2_bias_dim_0)};
     params[118].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_9_mlp_c_fc_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_9_mlp_c_fc_weight_dim_1)};
     params[119].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_9_mlp_c_fc_bias_dim_0)};
     params[120].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_9_mlp_c_proj_weight_dim_0), ParamDim(2048, 2048, &transformer_resblocks_9_mlp_c_proj_weight_dim_1)};
     params[121].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_9_mlp_c_proj_bias_dim_0)};
     params[122].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_10_ln_1_weight_dim_0)};
     params[123].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_10_ln_1_bias_dim_0)};
     params[124].shape_ptrs = {ParamDim(1536, 1536, &transformer_resblocks_10_attn_qkv_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_10_attn_qkv_weight_dim_1)};
     params[125].shape_ptrs = {ParamDim(2, 2, &transformer_resblocks_10_attn_cu_length_dim_0)};
     params[126].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_10_attn_proj_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_10_attn_proj_weight_dim_1)};
     params[127].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_10_attn_proj_bias_dim_0)};
     params[128].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_10_ln_2_weight_dim_0)};
     params[129].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_10_ln_2_bias_dim_0)};
     params[130].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_10_mlp_c_fc_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_10_mlp_c_fc_weight_dim_1)};
     params[131].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_10_mlp_c_fc_bias_dim_0)};
     params[132].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_10_mlp_c_proj_weight_dim_0), ParamDim(2048, 2048, &transformer_resblocks_10_mlp_c_proj_weight_dim_1)};
     params[133].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_10_mlp_c_proj_bias_dim_0)};
     params[134].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_11_ln_1_weight_dim_0)};
     params[135].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_11_ln_1_bias_dim_0)};
     params[136].shape_ptrs = {ParamDim(1536, 1536, &transformer_resblocks_11_attn_qkv_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_11_attn_qkv_weight_dim_1)};
     params[137].shape_ptrs = {ParamDim(2, 2, &transformer_resblocks_11_attn_cu_length_dim_0)};
     params[138].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_11_attn_proj_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_11_attn_proj_weight_dim_1)};
     params[139].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_11_attn_proj_bias_dim_0)};
     params[140].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_11_ln_2_weight_dim_0)};
     params[141].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_11_ln_2_bias_dim_0)};
     params[142].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_11_mlp_c_fc_weight_dim_0), ParamDim(512, 512, &transformer_resblocks_11_mlp_c_fc_weight_dim_1)};
     params[143].shape_ptrs = {ParamDim(2048, 2048, &transformer_resblocks_11_mlp_c_fc_bias_dim_0)};
     params[144].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_11_mlp_c_proj_weight_dim_0), ParamDim(2048, 2048, &transformer_resblocks_11_mlp_c_proj_weight_dim_1)};
     params[145].shape_ptrs = {ParamDim(512, 512, &transformer_resblocks_11_mlp_c_proj_bias_dim_0)};
     params[1].shape_ptrs = {ParamDim(1, 1, &reshape_167_0_dim_1), ParamDim(77, 77, &reshape_167_0_dim_0), ParamDim(512, 512, &transformer_resblocks_11_mlp_c_proj_weight_dim_0)};
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
             input = static_cast<decltype(input)>(params[0].ptr);

if (input == nullptr) {
    throw std::runtime_error("Constant input was not set! Set the value with set_constant.");
}
    
     reshape_0_0 = input;

if (transformer_resblocks_0_ln_1_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_0_ln_1_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_0_ln_1_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_0_ln_1_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_0_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_0_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_0_attn_cu_length == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_0_attn_cu_length was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_0_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_0_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_0_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_0_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_0_ln_2_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_0_ln_2_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_0_ln_2_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_0_ln_2_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_0_mlp_c_fc_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_0_mlp_c_fc_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_0_mlp_c_fc_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_0_mlp_c_fc_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_0_mlp_c_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_0_mlp_c_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_0_mlp_c_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_0_mlp_c_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_1_ln_1_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_1_ln_1_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_1_ln_1_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_1_ln_1_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_1_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_1_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_1_attn_cu_length == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_1_attn_cu_length was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_1_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_1_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_1_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_1_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_1_ln_2_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_1_ln_2_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_1_ln_2_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_1_ln_2_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_1_mlp_c_fc_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_1_mlp_c_fc_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_1_mlp_c_fc_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_1_mlp_c_fc_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_1_mlp_c_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_1_mlp_c_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_1_mlp_c_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_1_mlp_c_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_2_ln_1_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_2_ln_1_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_2_ln_1_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_2_ln_1_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_2_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_2_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_2_attn_cu_length == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_2_attn_cu_length was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_2_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_2_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_2_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_2_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_2_ln_2_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_2_ln_2_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_2_ln_2_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_2_ln_2_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_2_mlp_c_fc_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_2_mlp_c_fc_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_2_mlp_c_fc_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_2_mlp_c_fc_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_2_mlp_c_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_2_mlp_c_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_2_mlp_c_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_2_mlp_c_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_3_ln_1_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_3_ln_1_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_3_ln_1_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_3_ln_1_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_3_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_3_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_3_attn_cu_length == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_3_attn_cu_length was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_3_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_3_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_3_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_3_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_3_ln_2_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_3_ln_2_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_3_ln_2_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_3_ln_2_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_3_mlp_c_fc_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_3_mlp_c_fc_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_3_mlp_c_fc_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_3_mlp_c_fc_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_3_mlp_c_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_3_mlp_c_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_3_mlp_c_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_3_mlp_c_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_4_ln_1_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_4_ln_1_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_4_ln_1_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_4_ln_1_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_4_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_4_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_4_attn_cu_length == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_4_attn_cu_length was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_4_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_4_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_4_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_4_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_4_ln_2_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_4_ln_2_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_4_ln_2_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_4_ln_2_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_4_mlp_c_fc_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_4_mlp_c_fc_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_4_mlp_c_fc_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_4_mlp_c_fc_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_4_mlp_c_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_4_mlp_c_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_4_mlp_c_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_4_mlp_c_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_5_ln_1_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_5_ln_1_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_5_ln_1_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_5_ln_1_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_5_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_5_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_5_attn_cu_length == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_5_attn_cu_length was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_5_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_5_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_5_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_5_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_5_ln_2_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_5_ln_2_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_5_ln_2_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_5_ln_2_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_5_mlp_c_fc_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_5_mlp_c_fc_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_5_mlp_c_fc_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_5_mlp_c_fc_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_5_mlp_c_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_5_mlp_c_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_5_mlp_c_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_5_mlp_c_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_6_ln_1_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_6_ln_1_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_6_ln_1_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_6_ln_1_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_6_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_6_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_6_attn_cu_length == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_6_attn_cu_length was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_6_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_6_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_6_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_6_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_6_ln_2_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_6_ln_2_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_6_ln_2_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_6_ln_2_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_6_mlp_c_fc_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_6_mlp_c_fc_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_6_mlp_c_fc_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_6_mlp_c_fc_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_6_mlp_c_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_6_mlp_c_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_6_mlp_c_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_6_mlp_c_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_7_ln_1_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_7_ln_1_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_7_ln_1_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_7_ln_1_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_7_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_7_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_7_attn_cu_length == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_7_attn_cu_length was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_7_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_7_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_7_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_7_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_7_ln_2_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_7_ln_2_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_7_ln_2_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_7_ln_2_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_7_mlp_c_fc_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_7_mlp_c_fc_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_7_mlp_c_fc_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_7_mlp_c_fc_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_7_mlp_c_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_7_mlp_c_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_7_mlp_c_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_7_mlp_c_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_8_ln_1_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_8_ln_1_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_8_ln_1_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_8_ln_1_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_8_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_8_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_8_attn_cu_length == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_8_attn_cu_length was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_8_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_8_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_8_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_8_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_8_ln_2_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_8_ln_2_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_8_ln_2_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_8_ln_2_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_8_mlp_c_fc_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_8_mlp_c_fc_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_8_mlp_c_fc_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_8_mlp_c_fc_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_8_mlp_c_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_8_mlp_c_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_8_mlp_c_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_8_mlp_c_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_9_ln_1_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_9_ln_1_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_9_ln_1_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_9_ln_1_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_9_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_9_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_9_attn_cu_length == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_9_attn_cu_length was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_9_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_9_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_9_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_9_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_9_ln_2_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_9_ln_2_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_9_ln_2_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_9_ln_2_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_9_mlp_c_fc_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_9_mlp_c_fc_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_9_mlp_c_fc_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_9_mlp_c_fc_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_9_mlp_c_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_9_mlp_c_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_9_mlp_c_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_9_mlp_c_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_10_ln_1_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_10_ln_1_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_10_ln_1_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_10_ln_1_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_10_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_10_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_10_attn_cu_length == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_10_attn_cu_length was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_10_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_10_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_10_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_10_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_10_ln_2_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_10_ln_2_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_10_ln_2_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_10_ln_2_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_10_mlp_c_fc_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_10_mlp_c_fc_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_10_mlp_c_fc_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_10_mlp_c_fc_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_10_mlp_c_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_10_mlp_c_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_10_mlp_c_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_10_mlp_c_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_11_ln_1_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_11_ln_1_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_11_ln_1_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_11_ln_1_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_11_attn_qkv_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_11_attn_qkv_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_11_attn_cu_length == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_11_attn_cu_length was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_11_attn_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_11_attn_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_11_attn_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_11_attn_proj_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_11_ln_2_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_11_ln_2_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_11_ln_2_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_11_ln_2_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_11_mlp_c_fc_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_11_mlp_c_fc_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_11_mlp_c_fc_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_11_mlp_c_fc_bias was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_11_mlp_c_proj_weight == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_11_mlp_c_proj_weight was not set! Set the value with set_constant.");
}
    

if (transformer_resblocks_11_mlp_c_proj_bias == nullptr) {
    throw std::runtime_error("Constant transformer_resblocks_11_mlp_c_proj_bias was not set! Set the value with set_constant.");
}
    
     output_0 = static_cast<decltype(output_0)>(params[1].ptr);

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
        &input_dim_0,
        &input_dim_1,
        &reshape_0_0_dim_0
    );
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    batch_gather_1(
       reinterpret_cast<half*>(
        &(batch_gather_1_0->raw())), reinterpret_cast<half*>(
        &(token_embedding_weight->raw())), reinterpret_cast<int64_t*>(reshape_0_0),
        1,
        77,
        512,
        49408,
        global_workspace, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_176_n_elements = 1 * 77 * 512;
        invoke_fused_elementwise_176(reinterpret_cast<half*>(elementwise_4_0), reinterpret_cast<half*>(batch_gather_1_0),reinterpret_cast<half*>(positional_embedding),  fused_elementwise_176_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    permute102_5(
        (cutlass::half_t*)elementwise_4_0,
        (cutlass::half_t*)permute102_5_0,
        &reshape_3_0_dim_0,
        &reshape_3_0_dim_1,
        &reshape_3_0_dim_2,
        &reshape_3_0_dim_1,
        &reshape_3_0_dim_0,
        &reshape_3_0_dim_2,
        stream
    );
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_3_0_dim_1;

        M *= reshape_3_0_dim_0;

    

        int64_t N = 1;

        N *= reshape_3_0_dim_2;

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_6_0->raw())), reinterpret_cast<half*>(&(permute102_5_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_0_ln_1_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_0_ln_1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_177_n_elements = 77 * 1 * 512;
        invoke_fused_elementwise_177(reinterpret_cast<half*>(elementwise_7_0), reinterpret_cast<half*>(layernorm_6_0),  fused_elementwise_177_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_8(
        elementwise_7_0,
        transformer_resblocks_0_attn_qkv_weight,

        reshape_9_0,
        global_workspace,
        1,

        &reshape_3_0_dim_1,

        &reshape_3_0_dim_0,

        &reshape_3_0_dim_2,


        &transformer_resblocks_0_attn_qkv_weight_dim_0,

        &transformer_resblocks_0_attn_qkv_weight_dim_1,


        &reshape_3_0_dim_1,

        &reshape_3_0_dim_0,

        &transformer_resblocks_0_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_10_0->raw())), reinterpret_cast<half*>(&(reshape_9_0->raw())), reinterpret_cast<int*>(transformer_resblocks_0_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1024 * sizeof(float)),
        1,
        128,
        8,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_10_0,
        transformer_resblocks_0_attn_proj_weight,
        transformer_resblocks_0_attn_proj_bias,
        permute102_5_0,

        reshape_13_0,
        global_workspace,

     1,


        &reshape_11_0_dim_0,

        &reshape_11_0_dim_1,


        &transformer_resblocks_0_attn_proj_weight_dim_0,

        &transformer_resblocks_0_attn_proj_weight_dim_1,


        &reshape_11_0_dim_0,

        &transformer_resblocks_0_attn_proj_weight_dim_0,

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

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_14_0->raw())), reinterpret_cast<half*>(&(reshape_13_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_0_ln_2_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_0_ln_2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_14_0,
        transformer_resblocks_0_mlp_c_fc_weight,

        transformer_resblocks_0_mlp_c_fc_bias,

        gemm_rcr_bias_15_0,
        global_workspace,
        1,

        &reshape_13_0_dim_0,

        &reshape_13_0_dim_1,

        &reshape_13_0_dim_2,


        &transformer_resblocks_0_mlp_c_fc_weight_dim_0,

        &transformer_resblocks_0_mlp_c_fc_weight_dim_1,


        &reshape_13_0_dim_0,

        &reshape_13_0_dim_1,

        &transformer_resblocks_0_mlp_c_fc_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_178_n_elements = 77 * 1 * 2048;
        invoke_fused_elementwise_178(reinterpret_cast<half*>(elementwise_18_0), reinterpret_cast<half*>(gemm_rcr_bias_15_0),  fused_elementwise_178_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_19(
        elementwise_18_0,
        transformer_resblocks_0_mlp_c_proj_weight,

        transformer_resblocks_0_mlp_c_proj_bias,

        gemm_rcr_bias_19_0,
        global_workspace,
        1,

        &reshape_13_0_dim_0,

        &reshape_13_0_dim_1,

        &transformer_resblocks_0_mlp_c_fc_weight_dim_0,


        &transformer_resblocks_0_mlp_c_proj_weight_dim_0,

        &transformer_resblocks_0_mlp_c_proj_weight_dim_1,


        &reshape_13_0_dim_0,

        &reshape_13_0_dim_1,

        &transformer_resblocks_0_mlp_c_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_13_0_dim_0;

        M *= reshape_13_0_dim_1;

    

        int64_t N = 1;

        N *= transformer_resblocks_0_mlp_c_proj_weight_dim_0;

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_20_0->raw())), reinterpret_cast<half*>(&(gemm_rcr_bias_19_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_1_ln_1_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_1_ln_1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_177_n_elements = 77 * 1 * 512;
        invoke_fused_elementwise_177(reinterpret_cast<half*>(elementwise_21_0), reinterpret_cast<half*>(layernorm_20_0),  fused_elementwise_177_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_8(
        elementwise_21_0,
        transformer_resblocks_1_attn_qkv_weight,

        reshape_23_0,
        global_workspace,
        1,

        &reshape_13_0_dim_0,

        &reshape_13_0_dim_1,

        &transformer_resblocks_0_mlp_c_proj_weight_dim_0,


        &transformer_resblocks_1_attn_qkv_weight_dim_0,

        &transformer_resblocks_1_attn_qkv_weight_dim_1,


        &reshape_13_0_dim_0,

        &reshape_13_0_dim_1,

        &transformer_resblocks_1_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_24_0->raw())), reinterpret_cast<half*>(&(reshape_23_0->raw())), reinterpret_cast<int*>(transformer_resblocks_1_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1024 * sizeof(float)),
        1,
        128,
        8,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_24_0,
        transformer_resblocks_1_attn_proj_weight,
        transformer_resblocks_1_attn_proj_bias,
        gemm_rcr_bias_19_0,

        reshape_27_0,
        global_workspace,

     1,


        &reshape_25_0_dim_0,

        &reshape_25_0_dim_1,


        &transformer_resblocks_1_attn_proj_weight_dim_0,

        &transformer_resblocks_1_attn_proj_weight_dim_1,


        &reshape_25_0_dim_0,

        &transformer_resblocks_1_attn_proj_weight_dim_0,

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

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_28_0->raw())), reinterpret_cast<half*>(&(reshape_27_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_1_ln_2_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_1_ln_2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_28_0,
        transformer_resblocks_1_mlp_c_fc_weight,

        transformer_resblocks_1_mlp_c_fc_bias,

        gemm_rcr_bias_29_0,
        global_workspace,
        1,

        &reshape_27_0_dim_0,

        &reshape_27_0_dim_1,

        &reshape_27_0_dim_2,


        &transformer_resblocks_1_mlp_c_fc_weight_dim_0,

        &transformer_resblocks_1_mlp_c_fc_weight_dim_1,


        &reshape_27_0_dim_0,

        &reshape_27_0_dim_1,

        &transformer_resblocks_1_mlp_c_fc_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_180_n_elements = 77 * 1 * 2048;
        invoke_fused_elementwise_180(reinterpret_cast<half*>(elementwise_32_0), reinterpret_cast<half*>(gemm_rcr_bias_29_0),  fused_elementwise_180_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_19(
        elementwise_32_0,
        transformer_resblocks_1_mlp_c_proj_weight,

        transformer_resblocks_1_mlp_c_proj_bias,

        gemm_rcr_bias_33_0,
        global_workspace,
        1,

        &reshape_27_0_dim_0,

        &reshape_27_0_dim_1,

        &transformer_resblocks_1_mlp_c_fc_weight_dim_0,


        &transformer_resblocks_1_mlp_c_proj_weight_dim_0,

        &transformer_resblocks_1_mlp_c_proj_weight_dim_1,


        &reshape_27_0_dim_0,

        &reshape_27_0_dim_1,

        &transformer_resblocks_1_mlp_c_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_27_0_dim_0;

        M *= reshape_27_0_dim_1;

    

        int64_t N = 1;

        N *= transformer_resblocks_1_mlp_c_proj_weight_dim_0;

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_34_0->raw())), reinterpret_cast<half*>(&(gemm_rcr_bias_33_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_2_ln_1_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_2_ln_1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_177_n_elements = 77 * 1 * 512;
        invoke_fused_elementwise_177(reinterpret_cast<half*>(elementwise_35_0), reinterpret_cast<half*>(layernorm_34_0),  fused_elementwise_177_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_8(
        elementwise_35_0,
        transformer_resblocks_2_attn_qkv_weight,

        reshape_37_0,
        global_workspace,
        1,

        &reshape_27_0_dim_0,

        &reshape_27_0_dim_1,

        &transformer_resblocks_1_mlp_c_proj_weight_dim_0,


        &transformer_resblocks_2_attn_qkv_weight_dim_0,

        &transformer_resblocks_2_attn_qkv_weight_dim_1,


        &reshape_27_0_dim_0,

        &reshape_27_0_dim_1,

        &transformer_resblocks_2_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_38_0->raw())), reinterpret_cast<half*>(&(reshape_37_0->raw())), reinterpret_cast<int*>(transformer_resblocks_2_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1024 * sizeof(float)),
        1,
        128,
        8,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_38_0,
        transformer_resblocks_2_attn_proj_weight,
        transformer_resblocks_2_attn_proj_bias,
        gemm_rcr_bias_33_0,

        reshape_41_0,
        global_workspace,

     1,


        &reshape_39_0_dim_0,

        &reshape_39_0_dim_1,


        &transformer_resblocks_2_attn_proj_weight_dim_0,

        &transformer_resblocks_2_attn_proj_weight_dim_1,


        &reshape_39_0_dim_0,

        &transformer_resblocks_2_attn_proj_weight_dim_0,

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

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_42_0->raw())), reinterpret_cast<half*>(&(reshape_41_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_2_ln_2_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_2_ln_2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_42_0,
        transformer_resblocks_2_mlp_c_fc_weight,

        transformer_resblocks_2_mlp_c_fc_bias,

        gemm_rcr_bias_43_0,
        global_workspace,
        1,

        &reshape_41_0_dim_0,

        &reshape_41_0_dim_1,

        &reshape_41_0_dim_2,


        &transformer_resblocks_2_mlp_c_fc_weight_dim_0,

        &transformer_resblocks_2_mlp_c_fc_weight_dim_1,


        &reshape_41_0_dim_0,

        &reshape_41_0_dim_1,

        &transformer_resblocks_2_mlp_c_fc_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_182_n_elements = 77 * 1 * 2048;
        invoke_fused_elementwise_182(reinterpret_cast<half*>(elementwise_46_0), reinterpret_cast<half*>(gemm_rcr_bias_43_0),  fused_elementwise_182_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_19(
        elementwise_46_0,
        transformer_resblocks_2_mlp_c_proj_weight,

        transformer_resblocks_2_mlp_c_proj_bias,

        gemm_rcr_bias_47_0,
        global_workspace,
        1,

        &reshape_41_0_dim_0,

        &reshape_41_0_dim_1,

        &transformer_resblocks_2_mlp_c_fc_weight_dim_0,


        &transformer_resblocks_2_mlp_c_proj_weight_dim_0,

        &transformer_resblocks_2_mlp_c_proj_weight_dim_1,


        &reshape_41_0_dim_0,

        &reshape_41_0_dim_1,

        &transformer_resblocks_2_mlp_c_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_41_0_dim_0;

        M *= reshape_41_0_dim_1;

    

        int64_t N = 1;

        N *= transformer_resblocks_2_mlp_c_proj_weight_dim_0;

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_48_0->raw())), reinterpret_cast<half*>(&(gemm_rcr_bias_47_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_3_ln_1_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_3_ln_1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_177_n_elements = 77 * 1 * 512;
        invoke_fused_elementwise_177(reinterpret_cast<half*>(elementwise_49_0), reinterpret_cast<half*>(layernorm_48_0),  fused_elementwise_177_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_8(
        elementwise_49_0,
        transformer_resblocks_3_attn_qkv_weight,

        reshape_51_0,
        global_workspace,
        1,

        &reshape_41_0_dim_0,

        &reshape_41_0_dim_1,

        &transformer_resblocks_2_mlp_c_proj_weight_dim_0,


        &transformer_resblocks_3_attn_qkv_weight_dim_0,

        &transformer_resblocks_3_attn_qkv_weight_dim_1,


        &reshape_41_0_dim_0,

        &reshape_41_0_dim_1,

        &transformer_resblocks_3_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_52_0->raw())), reinterpret_cast<half*>(&(reshape_51_0->raw())), reinterpret_cast<int*>(transformer_resblocks_3_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1024 * sizeof(float)),
        1,
        128,
        8,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_52_0,
        transformer_resblocks_3_attn_proj_weight,
        transformer_resblocks_3_attn_proj_bias,
        gemm_rcr_bias_47_0,

        reshape_55_0,
        global_workspace,

     1,


        &reshape_53_0_dim_0,

        &reshape_53_0_dim_1,


        &transformer_resblocks_3_attn_proj_weight_dim_0,

        &transformer_resblocks_3_attn_proj_weight_dim_1,


        &reshape_53_0_dim_0,

        &transformer_resblocks_3_attn_proj_weight_dim_0,

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

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_56_0->raw())), reinterpret_cast<half*>(&(reshape_55_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_3_ln_2_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_3_ln_2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_56_0,
        transformer_resblocks_3_mlp_c_fc_weight,

        transformer_resblocks_3_mlp_c_fc_bias,

        gemm_rcr_bias_57_0,
        global_workspace,
        1,

        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &reshape_55_0_dim_2,


        &transformer_resblocks_3_mlp_c_fc_weight_dim_0,

        &transformer_resblocks_3_mlp_c_fc_weight_dim_1,


        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &transformer_resblocks_3_mlp_c_fc_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_184_n_elements = 77 * 1 * 2048;
        invoke_fused_elementwise_184(reinterpret_cast<half*>(elementwise_60_0), reinterpret_cast<half*>(gemm_rcr_bias_57_0),  fused_elementwise_184_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_19(
        elementwise_60_0,
        transformer_resblocks_3_mlp_c_proj_weight,

        transformer_resblocks_3_mlp_c_proj_bias,

        gemm_rcr_bias_61_0,
        global_workspace,
        1,

        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &transformer_resblocks_3_mlp_c_fc_weight_dim_0,


        &transformer_resblocks_3_mlp_c_proj_weight_dim_0,

        &transformer_resblocks_3_mlp_c_proj_weight_dim_1,


        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &transformer_resblocks_3_mlp_c_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_55_0_dim_0;

        M *= reshape_55_0_dim_1;

    

        int64_t N = 1;

        N *= transformer_resblocks_3_mlp_c_proj_weight_dim_0;

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_62_0->raw())), reinterpret_cast<half*>(&(gemm_rcr_bias_61_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_4_ln_1_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_4_ln_1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_177_n_elements = 77 * 1 * 512;
        invoke_fused_elementwise_177(reinterpret_cast<half*>(elementwise_63_0), reinterpret_cast<half*>(layernorm_62_0),  fused_elementwise_177_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_8(
        elementwise_63_0,
        transformer_resblocks_4_attn_qkv_weight,

        reshape_65_0,
        global_workspace,
        1,

        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &transformer_resblocks_3_mlp_c_proj_weight_dim_0,


        &transformer_resblocks_4_attn_qkv_weight_dim_0,

        &transformer_resblocks_4_attn_qkv_weight_dim_1,


        &reshape_55_0_dim_0,

        &reshape_55_0_dim_1,

        &transformer_resblocks_4_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_66_0->raw())), reinterpret_cast<half*>(&(reshape_65_0->raw())), reinterpret_cast<int*>(transformer_resblocks_4_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1024 * sizeof(float)),
        1,
        128,
        8,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_66_0,
        transformer_resblocks_4_attn_proj_weight,
        transformer_resblocks_4_attn_proj_bias,
        gemm_rcr_bias_61_0,

        reshape_69_0,
        global_workspace,

     1,


        &reshape_67_0_dim_0,

        &reshape_67_0_dim_1,


        &transformer_resblocks_4_attn_proj_weight_dim_0,

        &transformer_resblocks_4_attn_proj_weight_dim_1,


        &reshape_67_0_dim_0,

        &transformer_resblocks_4_attn_proj_weight_dim_0,

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

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_70_0->raw())), reinterpret_cast<half*>(&(reshape_69_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_4_ln_2_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_4_ln_2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_70_0,
        transformer_resblocks_4_mlp_c_fc_weight,

        transformer_resblocks_4_mlp_c_fc_bias,

        gemm_rcr_bias_71_0,
        global_workspace,
        1,

        &reshape_69_0_dim_0,

        &reshape_69_0_dim_1,

        &reshape_69_0_dim_2,


        &transformer_resblocks_4_mlp_c_fc_weight_dim_0,

        &transformer_resblocks_4_mlp_c_fc_weight_dim_1,


        &reshape_69_0_dim_0,

        &reshape_69_0_dim_1,

        &transformer_resblocks_4_mlp_c_fc_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_186_n_elements = 77 * 1 * 2048;
        invoke_fused_elementwise_186(reinterpret_cast<half*>(elementwise_74_0), reinterpret_cast<half*>(gemm_rcr_bias_71_0),  fused_elementwise_186_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_19(
        elementwise_74_0,
        transformer_resblocks_4_mlp_c_proj_weight,

        transformer_resblocks_4_mlp_c_proj_bias,

        gemm_rcr_bias_75_0,
        global_workspace,
        1,

        &reshape_69_0_dim_0,

        &reshape_69_0_dim_1,

        &transformer_resblocks_4_mlp_c_fc_weight_dim_0,


        &transformer_resblocks_4_mlp_c_proj_weight_dim_0,

        &transformer_resblocks_4_mlp_c_proj_weight_dim_1,


        &reshape_69_0_dim_0,

        &reshape_69_0_dim_1,

        &transformer_resblocks_4_mlp_c_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_69_0_dim_0;

        M *= reshape_69_0_dim_1;

    

        int64_t N = 1;

        N *= transformer_resblocks_4_mlp_c_proj_weight_dim_0;

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_76_0->raw())), reinterpret_cast<half*>(&(gemm_rcr_bias_75_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_5_ln_1_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_5_ln_1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_177_n_elements = 77 * 1 * 512;
        invoke_fused_elementwise_177(reinterpret_cast<half*>(elementwise_77_0), reinterpret_cast<half*>(layernorm_76_0),  fused_elementwise_177_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_8(
        elementwise_77_0,
        transformer_resblocks_5_attn_qkv_weight,

        reshape_79_0,
        global_workspace,
        1,

        &reshape_69_0_dim_0,

        &reshape_69_0_dim_1,

        &transformer_resblocks_4_mlp_c_proj_weight_dim_0,


        &transformer_resblocks_5_attn_qkv_weight_dim_0,

        &transformer_resblocks_5_attn_qkv_weight_dim_1,


        &reshape_69_0_dim_0,

        &reshape_69_0_dim_1,

        &transformer_resblocks_5_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_80_0->raw())), reinterpret_cast<half*>(&(reshape_79_0->raw())), reinterpret_cast<int*>(transformer_resblocks_5_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1024 * sizeof(float)),
        1,
        128,
        8,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_80_0,
        transformer_resblocks_5_attn_proj_weight,
        transformer_resblocks_5_attn_proj_bias,
        gemm_rcr_bias_75_0,

        reshape_83_0,
        global_workspace,

     1,


        &reshape_81_0_dim_0,

        &reshape_81_0_dim_1,


        &transformer_resblocks_5_attn_proj_weight_dim_0,

        &transformer_resblocks_5_attn_proj_weight_dim_1,


        &reshape_81_0_dim_0,

        &transformer_resblocks_5_attn_proj_weight_dim_0,

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

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_84_0->raw())), reinterpret_cast<half*>(&(reshape_83_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_5_ln_2_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_5_ln_2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_84_0,
        transformer_resblocks_5_mlp_c_fc_weight,

        transformer_resblocks_5_mlp_c_fc_bias,

        gemm_rcr_bias_85_0,
        global_workspace,
        1,

        &reshape_83_0_dim_0,

        &reshape_83_0_dim_1,

        &reshape_83_0_dim_2,


        &transformer_resblocks_5_mlp_c_fc_weight_dim_0,

        &transformer_resblocks_5_mlp_c_fc_weight_dim_1,


        &reshape_83_0_dim_0,

        &reshape_83_0_dim_1,

        &transformer_resblocks_5_mlp_c_fc_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_188_n_elements = 77 * 1 * 2048;
        invoke_fused_elementwise_188(reinterpret_cast<half*>(elementwise_88_0), reinterpret_cast<half*>(gemm_rcr_bias_85_0),  fused_elementwise_188_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_19(
        elementwise_88_0,
        transformer_resblocks_5_mlp_c_proj_weight,

        transformer_resblocks_5_mlp_c_proj_bias,

        gemm_rcr_bias_89_0,
        global_workspace,
        1,

        &reshape_83_0_dim_0,

        &reshape_83_0_dim_1,

        &transformer_resblocks_5_mlp_c_fc_weight_dim_0,


        &transformer_resblocks_5_mlp_c_proj_weight_dim_0,

        &transformer_resblocks_5_mlp_c_proj_weight_dim_1,


        &reshape_83_0_dim_0,

        &reshape_83_0_dim_1,

        &transformer_resblocks_5_mlp_c_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_83_0_dim_0;

        M *= reshape_83_0_dim_1;

    

        int64_t N = 1;

        N *= transformer_resblocks_5_mlp_c_proj_weight_dim_0;

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_90_0->raw())), reinterpret_cast<half*>(&(gemm_rcr_bias_89_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_6_ln_1_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_6_ln_1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_177_n_elements = 77 * 1 * 512;
        invoke_fused_elementwise_177(reinterpret_cast<half*>(elementwise_91_0), reinterpret_cast<half*>(layernorm_90_0),  fused_elementwise_177_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_8(
        elementwise_91_0,
        transformer_resblocks_6_attn_qkv_weight,

        reshape_93_0,
        global_workspace,
        1,

        &reshape_83_0_dim_0,

        &reshape_83_0_dim_1,

        &transformer_resblocks_5_mlp_c_proj_weight_dim_0,


        &transformer_resblocks_6_attn_qkv_weight_dim_0,

        &transformer_resblocks_6_attn_qkv_weight_dim_1,


        &reshape_83_0_dim_0,

        &reshape_83_0_dim_1,

        &transformer_resblocks_6_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_94_0->raw())), reinterpret_cast<half*>(&(reshape_93_0->raw())), reinterpret_cast<int*>(transformer_resblocks_6_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1024 * sizeof(float)),
        1,
        128,
        8,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_94_0,
        transformer_resblocks_6_attn_proj_weight,
        transformer_resblocks_6_attn_proj_bias,
        gemm_rcr_bias_89_0,

        reshape_97_0,
        global_workspace,

     1,


        &reshape_95_0_dim_0,

        &reshape_95_0_dim_1,


        &transformer_resblocks_6_attn_proj_weight_dim_0,

        &transformer_resblocks_6_attn_proj_weight_dim_1,


        &reshape_95_0_dim_0,

        &transformer_resblocks_6_attn_proj_weight_dim_0,

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

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_98_0->raw())), reinterpret_cast<half*>(&(reshape_97_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_6_ln_2_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_6_ln_2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_98_0,
        transformer_resblocks_6_mlp_c_fc_weight,

        transformer_resblocks_6_mlp_c_fc_bias,

        gemm_rcr_bias_99_0,
        global_workspace,
        1,

        &reshape_97_0_dim_0,

        &reshape_97_0_dim_1,

        &reshape_97_0_dim_2,


        &transformer_resblocks_6_mlp_c_fc_weight_dim_0,

        &transformer_resblocks_6_mlp_c_fc_weight_dim_1,


        &reshape_97_0_dim_0,

        &reshape_97_0_dim_1,

        &transformer_resblocks_6_mlp_c_fc_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_190_n_elements = 77 * 1 * 2048;
        invoke_fused_elementwise_190(reinterpret_cast<half*>(elementwise_102_0), reinterpret_cast<half*>(gemm_rcr_bias_99_0),  fused_elementwise_190_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_19(
        elementwise_102_0,
        transformer_resblocks_6_mlp_c_proj_weight,

        transformer_resblocks_6_mlp_c_proj_bias,

        gemm_rcr_bias_103_0,
        global_workspace,
        1,

        &reshape_97_0_dim_0,

        &reshape_97_0_dim_1,

        &transformer_resblocks_6_mlp_c_fc_weight_dim_0,


        &transformer_resblocks_6_mlp_c_proj_weight_dim_0,

        &transformer_resblocks_6_mlp_c_proj_weight_dim_1,


        &reshape_97_0_dim_0,

        &reshape_97_0_dim_1,

        &transformer_resblocks_6_mlp_c_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_97_0_dim_0;

        M *= reshape_97_0_dim_1;

    

        int64_t N = 1;

        N *= transformer_resblocks_6_mlp_c_proj_weight_dim_0;

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_104_0->raw())), reinterpret_cast<half*>(&(gemm_rcr_bias_103_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_7_ln_1_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_7_ln_1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_177_n_elements = 77 * 1 * 512;
        invoke_fused_elementwise_177(reinterpret_cast<half*>(elementwise_105_0), reinterpret_cast<half*>(layernorm_104_0),  fused_elementwise_177_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_8(
        elementwise_105_0,
        transformer_resblocks_7_attn_qkv_weight,

        reshape_107_0,
        global_workspace,
        1,

        &reshape_97_0_dim_0,

        &reshape_97_0_dim_1,

        &transformer_resblocks_6_mlp_c_proj_weight_dim_0,


        &transformer_resblocks_7_attn_qkv_weight_dim_0,

        &transformer_resblocks_7_attn_qkv_weight_dim_1,


        &reshape_97_0_dim_0,

        &reshape_97_0_dim_1,

        &transformer_resblocks_7_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_108_0->raw())), reinterpret_cast<half*>(&(reshape_107_0->raw())), reinterpret_cast<int*>(transformer_resblocks_7_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1024 * sizeof(float)),
        1,
        128,
        8,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_108_0,
        transformer_resblocks_7_attn_proj_weight,
        transformer_resblocks_7_attn_proj_bias,
        gemm_rcr_bias_103_0,

        reshape_111_0,
        global_workspace,

     1,


        &reshape_109_0_dim_0,

        &reshape_109_0_dim_1,


        &transformer_resblocks_7_attn_proj_weight_dim_0,

        &transformer_resblocks_7_attn_proj_weight_dim_1,


        &reshape_109_0_dim_0,

        &transformer_resblocks_7_attn_proj_weight_dim_0,

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

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_112_0->raw())), reinterpret_cast<half*>(&(reshape_111_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_7_ln_2_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_7_ln_2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_112_0,
        transformer_resblocks_7_mlp_c_fc_weight,

        transformer_resblocks_7_mlp_c_fc_bias,

        gemm_rcr_bias_113_0,
        global_workspace,
        1,

        &reshape_111_0_dim_0,

        &reshape_111_0_dim_1,

        &reshape_111_0_dim_2,


        &transformer_resblocks_7_mlp_c_fc_weight_dim_0,

        &transformer_resblocks_7_mlp_c_fc_weight_dim_1,


        &reshape_111_0_dim_0,

        &reshape_111_0_dim_1,

        &transformer_resblocks_7_mlp_c_fc_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_192_n_elements = 77 * 1 * 2048;
        invoke_fused_elementwise_192(reinterpret_cast<half*>(elementwise_116_0), reinterpret_cast<half*>(gemm_rcr_bias_113_0),  fused_elementwise_192_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_19(
        elementwise_116_0,
        transformer_resblocks_7_mlp_c_proj_weight,

        transformer_resblocks_7_mlp_c_proj_bias,

        gemm_rcr_bias_117_0,
        global_workspace,
        1,

        &reshape_111_0_dim_0,

        &reshape_111_0_dim_1,

        &transformer_resblocks_7_mlp_c_fc_weight_dim_0,


        &transformer_resblocks_7_mlp_c_proj_weight_dim_0,

        &transformer_resblocks_7_mlp_c_proj_weight_dim_1,


        &reshape_111_0_dim_0,

        &reshape_111_0_dim_1,

        &transformer_resblocks_7_mlp_c_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_111_0_dim_0;

        M *= reshape_111_0_dim_1;

    

        int64_t N = 1;

        N *= transformer_resblocks_7_mlp_c_proj_weight_dim_0;

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_118_0->raw())), reinterpret_cast<half*>(&(gemm_rcr_bias_117_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_8_ln_1_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_8_ln_1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_177_n_elements = 77 * 1 * 512;
        invoke_fused_elementwise_177(reinterpret_cast<half*>(elementwise_119_0), reinterpret_cast<half*>(layernorm_118_0),  fused_elementwise_177_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_8(
        elementwise_119_0,
        transformer_resblocks_8_attn_qkv_weight,

        reshape_121_0,
        global_workspace,
        1,

        &reshape_111_0_dim_0,

        &reshape_111_0_dim_1,

        &transformer_resblocks_7_mlp_c_proj_weight_dim_0,


        &transformer_resblocks_8_attn_qkv_weight_dim_0,

        &transformer_resblocks_8_attn_qkv_weight_dim_1,


        &reshape_111_0_dim_0,

        &reshape_111_0_dim_1,

        &transformer_resblocks_8_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_122_0->raw())), reinterpret_cast<half*>(&(reshape_121_0->raw())), reinterpret_cast<int*>(transformer_resblocks_8_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1024 * sizeof(float)),
        1,
        128,
        8,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_122_0,
        transformer_resblocks_8_attn_proj_weight,
        transformer_resblocks_8_attn_proj_bias,
        gemm_rcr_bias_117_0,

        reshape_125_0,
        global_workspace,

     1,


        &reshape_123_0_dim_0,

        &reshape_123_0_dim_1,


        &transformer_resblocks_8_attn_proj_weight_dim_0,

        &transformer_resblocks_8_attn_proj_weight_dim_1,


        &reshape_123_0_dim_0,

        &transformer_resblocks_8_attn_proj_weight_dim_0,

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

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_126_0->raw())), reinterpret_cast<half*>(&(reshape_125_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_8_ln_2_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_8_ln_2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_126_0,
        transformer_resblocks_8_mlp_c_fc_weight,

        transformer_resblocks_8_mlp_c_fc_bias,

        gemm_rcr_bias_127_0,
        global_workspace,
        1,

        &reshape_125_0_dim_0,

        &reshape_125_0_dim_1,

        &reshape_125_0_dim_2,


        &transformer_resblocks_8_mlp_c_fc_weight_dim_0,

        &transformer_resblocks_8_mlp_c_fc_weight_dim_1,


        &reshape_125_0_dim_0,

        &reshape_125_0_dim_1,

        &transformer_resblocks_8_mlp_c_fc_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_194_n_elements = 77 * 1 * 2048;
        invoke_fused_elementwise_194(reinterpret_cast<half*>(elementwise_130_0), reinterpret_cast<half*>(gemm_rcr_bias_127_0),  fused_elementwise_194_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_19(
        elementwise_130_0,
        transformer_resblocks_8_mlp_c_proj_weight,

        transformer_resblocks_8_mlp_c_proj_bias,

        gemm_rcr_bias_131_0,
        global_workspace,
        1,

        &reshape_125_0_dim_0,

        &reshape_125_0_dim_1,

        &transformer_resblocks_8_mlp_c_fc_weight_dim_0,


        &transformer_resblocks_8_mlp_c_proj_weight_dim_0,

        &transformer_resblocks_8_mlp_c_proj_weight_dim_1,


        &reshape_125_0_dim_0,

        &reshape_125_0_dim_1,

        &transformer_resblocks_8_mlp_c_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_125_0_dim_0;

        M *= reshape_125_0_dim_1;

    

        int64_t N = 1;

        N *= transformer_resblocks_8_mlp_c_proj_weight_dim_0;

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_132_0->raw())), reinterpret_cast<half*>(&(gemm_rcr_bias_131_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_9_ln_1_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_9_ln_1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_177_n_elements = 77 * 1 * 512;
        invoke_fused_elementwise_177(reinterpret_cast<half*>(elementwise_133_0), reinterpret_cast<half*>(layernorm_132_0),  fused_elementwise_177_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_8(
        elementwise_133_0,
        transformer_resblocks_9_attn_qkv_weight,

        reshape_135_0,
        global_workspace,
        1,

        &reshape_125_0_dim_0,

        &reshape_125_0_dim_1,

        &transformer_resblocks_8_mlp_c_proj_weight_dim_0,


        &transformer_resblocks_9_attn_qkv_weight_dim_0,

        &transformer_resblocks_9_attn_qkv_weight_dim_1,


        &reshape_125_0_dim_0,

        &reshape_125_0_dim_1,

        &transformer_resblocks_9_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_136_0->raw())), reinterpret_cast<half*>(&(reshape_135_0->raw())), reinterpret_cast<int*>(transformer_resblocks_9_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1024 * sizeof(float)),
        1,
        128,
        8,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_136_0,
        transformer_resblocks_9_attn_proj_weight,
        transformer_resblocks_9_attn_proj_bias,
        gemm_rcr_bias_131_0,

        reshape_139_0,
        global_workspace,

     1,


        &reshape_137_0_dim_0,

        &reshape_137_0_dim_1,


        &transformer_resblocks_9_attn_proj_weight_dim_0,

        &transformer_resblocks_9_attn_proj_weight_dim_1,


        &reshape_137_0_dim_0,

        &transformer_resblocks_9_attn_proj_weight_dim_0,

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

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_140_0->raw())), reinterpret_cast<half*>(&(reshape_139_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_9_ln_2_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_9_ln_2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_140_0,
        transformer_resblocks_9_mlp_c_fc_weight,

        transformer_resblocks_9_mlp_c_fc_bias,

        gemm_rcr_bias_141_0,
        global_workspace,
        1,

        &reshape_139_0_dim_0,

        &reshape_139_0_dim_1,

        &reshape_139_0_dim_2,


        &transformer_resblocks_9_mlp_c_fc_weight_dim_0,

        &transformer_resblocks_9_mlp_c_fc_weight_dim_1,


        &reshape_139_0_dim_0,

        &reshape_139_0_dim_1,

        &transformer_resblocks_9_mlp_c_fc_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_196_n_elements = 77 * 1 * 2048;
        invoke_fused_elementwise_196(reinterpret_cast<half*>(elementwise_144_0), reinterpret_cast<half*>(gemm_rcr_bias_141_0),  fused_elementwise_196_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_19(
        elementwise_144_0,
        transformer_resblocks_9_mlp_c_proj_weight,

        transformer_resblocks_9_mlp_c_proj_bias,

        gemm_rcr_bias_145_0,
        global_workspace,
        1,

        &reshape_139_0_dim_0,

        &reshape_139_0_dim_1,

        &transformer_resblocks_9_mlp_c_fc_weight_dim_0,


        &transformer_resblocks_9_mlp_c_proj_weight_dim_0,

        &transformer_resblocks_9_mlp_c_proj_weight_dim_1,


        &reshape_139_0_dim_0,

        &reshape_139_0_dim_1,

        &transformer_resblocks_9_mlp_c_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_139_0_dim_0;

        M *= reshape_139_0_dim_1;

    

        int64_t N = 1;

        N *= transformer_resblocks_9_mlp_c_proj_weight_dim_0;

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_146_0->raw())), reinterpret_cast<half*>(&(gemm_rcr_bias_145_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_10_ln_1_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_10_ln_1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_177_n_elements = 77 * 1 * 512;
        invoke_fused_elementwise_177(reinterpret_cast<half*>(elementwise_147_0), reinterpret_cast<half*>(layernorm_146_0),  fused_elementwise_177_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_8(
        elementwise_147_0,
        transformer_resblocks_10_attn_qkv_weight,

        reshape_149_0,
        global_workspace,
        1,

        &reshape_139_0_dim_0,

        &reshape_139_0_dim_1,

        &transformer_resblocks_9_mlp_c_proj_weight_dim_0,


        &transformer_resblocks_10_attn_qkv_weight_dim_0,

        &transformer_resblocks_10_attn_qkv_weight_dim_1,


        &reshape_139_0_dim_0,

        &reshape_139_0_dim_1,

        &transformer_resblocks_10_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_150_0->raw())), reinterpret_cast<half*>(&(reshape_149_0->raw())), reinterpret_cast<int*>(transformer_resblocks_10_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1024 * sizeof(float)),
        1,
        128,
        8,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_150_0,
        transformer_resblocks_10_attn_proj_weight,
        transformer_resblocks_10_attn_proj_bias,
        gemm_rcr_bias_145_0,

        reshape_153_0,
        global_workspace,

     1,


        &reshape_151_0_dim_0,

        &reshape_151_0_dim_1,


        &transformer_resblocks_10_attn_proj_weight_dim_0,

        &transformer_resblocks_10_attn_proj_weight_dim_1,


        &reshape_151_0_dim_0,

        &transformer_resblocks_10_attn_proj_weight_dim_0,

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

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_154_0->raw())), reinterpret_cast<half*>(&(reshape_153_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_10_ln_2_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_10_ln_2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_154_0,
        transformer_resblocks_10_mlp_c_fc_weight,

        transformer_resblocks_10_mlp_c_fc_bias,

        gemm_rcr_bias_155_0,
        global_workspace,
        1,

        &reshape_153_0_dim_0,

        &reshape_153_0_dim_1,

        &reshape_153_0_dim_2,


        &transformer_resblocks_10_mlp_c_fc_weight_dim_0,

        &transformer_resblocks_10_mlp_c_fc_weight_dim_1,


        &reshape_153_0_dim_0,

        &reshape_153_0_dim_1,

        &transformer_resblocks_10_mlp_c_fc_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_198_n_elements = 77 * 1 * 2048;
        invoke_fused_elementwise_198(reinterpret_cast<half*>(elementwise_158_0), reinterpret_cast<half*>(gemm_rcr_bias_155_0),  fused_elementwise_198_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_19(
        elementwise_158_0,
        transformer_resblocks_10_mlp_c_proj_weight,

        transformer_resblocks_10_mlp_c_proj_bias,

        gemm_rcr_bias_159_0,
        global_workspace,
        1,

        &reshape_153_0_dim_0,

        &reshape_153_0_dim_1,

        &transformer_resblocks_10_mlp_c_fc_weight_dim_0,


        &transformer_resblocks_10_mlp_c_proj_weight_dim_0,

        &transformer_resblocks_10_mlp_c_proj_weight_dim_1,


        &reshape_153_0_dim_0,

        &reshape_153_0_dim_1,

        &transformer_resblocks_10_mlp_c_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_153_0_dim_0;

        M *= reshape_153_0_dim_1;

    

        int64_t N = 1;

        N *= transformer_resblocks_10_mlp_c_proj_weight_dim_0;

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_160_0->raw())), reinterpret_cast<half*>(&(gemm_rcr_bias_159_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_11_ln_1_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_11_ln_1_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_177_n_elements = 77 * 1 * 512;
        invoke_fused_elementwise_177(reinterpret_cast<half*>(elementwise_161_0), reinterpret_cast<half*>(layernorm_160_0),  fused_elementwise_177_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_8(
        elementwise_161_0,
        transformer_resblocks_11_attn_qkv_weight,

        reshape_163_0,
        global_workspace,
        1,

        &reshape_153_0_dim_0,

        &reshape_153_0_dim_1,

        &transformer_resblocks_10_mlp_c_proj_weight_dim_0,


        &transformer_resblocks_11_attn_qkv_weight_dim_0,

        &transformer_resblocks_11_attn_qkv_weight_dim_1,


        &reshape_153_0_dim_0,

        &reshape_153_0_dim_1,

        &transformer_resblocks_11_attn_qkv_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    flash_attention_10(
       reinterpret_cast<half*>(&(flash_attention_164_0->raw())), reinterpret_cast<half*>(&(reshape_163_0->raw())), reinterpret_cast<int*>(transformer_resblocks_11_attn_cu_length),
        reinterpret_cast<float*>(global_workspace), reinterpret_cast<float*>(global_workspace + 1024 * sizeof(float)),
        1,
        128,
        8,
        64,
        0.0,
        0.125,
        true, false, stream /* default stream */
    );
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_add_12(
        flash_attention_164_0,
        transformer_resblocks_11_attn_proj_weight,
        transformer_resblocks_11_attn_proj_bias,
        gemm_rcr_bias_159_0,

        reshape_167_0,
        global_workspace,

     1,


        &reshape_165_0_dim_0,

        &reshape_165_0_dim_1,


        &transformer_resblocks_11_attn_proj_weight_dim_0,

        &transformer_resblocks_11_attn_proj_weight_dim_1,


        &reshape_165_0_dim_0,

        &transformer_resblocks_11_attn_proj_weight_dim_0,

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

    
        layernorm_6(
           reinterpret_cast<half*>(&(layernorm_168_0->raw())), reinterpret_cast<half*>(&(reshape_167_0->raw())), reinterpret_cast<half*>(&(transformer_resblocks_11_ln_2_weight->raw())), reinterpret_cast<half*>(&(transformer_resblocks_11_ln_2_bias->raw())),
           M, N, 1e-05, stream /* default stream */
        );
      }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_15(
        layernorm_168_0,
        transformer_resblocks_11_mlp_c_fc_weight,

        transformer_resblocks_11_mlp_c_fc_bias,

        gemm_rcr_bias_169_0,
        global_workspace,
        1,

        &reshape_167_0_dim_0,

        &reshape_167_0_dim_1,

        &reshape_167_0_dim_2,


        &transformer_resblocks_11_mlp_c_fc_weight_dim_0,

        &transformer_resblocks_11_mlp_c_fc_weight_dim_1,


        &reshape_167_0_dim_0,

        &reshape_167_0_dim_1,

        &transformer_resblocks_11_mlp_c_fc_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
        int fused_elementwise_200_n_elements = 77 * 1 * 2048;
        invoke_fused_elementwise_200(reinterpret_cast<half*>(elementwise_172_0), reinterpret_cast<half*>(gemm_rcr_bias_169_0),  fused_elementwise_200_n_elements, stream);
    }
    
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    {
    

    gemm_rcr_bias_19(
        elementwise_172_0,
        transformer_resblocks_11_mlp_c_proj_weight,

        transformer_resblocks_11_mlp_c_proj_bias,

        gemm_rcr_bias_173_0,
        global_workspace,
        1,

        &reshape_167_0_dim_0,

        &reshape_167_0_dim_1,

        &transformer_resblocks_11_mlp_c_fc_weight_dim_0,


        &transformer_resblocks_11_mlp_c_proj_weight_dim_0,

        &transformer_resblocks_11_mlp_c_proj_weight_dim_1,


        &reshape_167_0_dim_0,

        &reshape_167_0_dim_1,

        &transformer_resblocks_11_mlp_c_proj_weight_dim_0,

        stream
    );
    }
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
    permute102_174(
        (cutlass::half_t*)gemm_rcr_bias_173_0,
        (cutlass::half_t*)permute102_174_0,
        &reshape_167_0_dim_0,
        &reshape_167_0_dim_1,
        &transformer_resblocks_11_mlp_c_proj_weight_dim_0,
        &reshape_167_0_dim_1,
        &reshape_167_0_dim_0,
        &transformer_resblocks_11_mlp_c_proj_weight_dim_0,
        stream
    );
      DeviceCheckLastError(__FILE__, __LINE__);
  
  
      {
        
        int64_t M = 1;

        M *= reshape_167_0_dim_1;

        M *= reshape_167_0_dim_0;

    

        int64_t N = 1;

        N *= transformer_resblocks_11_mlp_c_proj_weight_dim_0;

    
        layernorm_175(
           reinterpret_cast<half*>(&(output_0->raw())), reinterpret_cast<half*>(&(permute102_174_0->raw())), reinterpret_cast<half*>(&(ln_final_weight->raw())), reinterpret_cast<half*>(&(ln_final_bias->raw())),
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

   int64_t* input {nullptr};
   int64_t* reshape_0_0 {nullptr};
   cutlass::half_t* token_embedding_weight {nullptr};
   cutlass::half_t* batch_gather_1_0 {nullptr};
   cutlass::half_t* size_2_0 {nullptr};
   cutlass::half_t* size_2_1 {nullptr};
   cutlass::half_t* positional_embedding {nullptr};
   cutlass::half_t* elementwise_4_0 {nullptr};
   cutlass::half_t* permute102_5_0 {nullptr};
   cutlass::half_t* transformer_resblocks_0_ln_1_weight {nullptr};
   cutlass::half_t* transformer_resblocks_0_ln_1_bias {nullptr};
   cutlass::half_t* layernorm_6_0 {nullptr};
   cutlass::half_t* elementwise_7_0 {nullptr};
   cutlass::half_t* transformer_resblocks_0_attn_qkv_weight {nullptr};
   cutlass::half_t* reshape_9_0 {nullptr};
   int32_t* transformer_resblocks_0_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_10_0 {nullptr};
   cutlass::half_t* transformer_resblocks_0_attn_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_0_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_13_0 {nullptr};
   cutlass::half_t* transformer_resblocks_0_ln_2_weight {nullptr};
   cutlass::half_t* transformer_resblocks_0_ln_2_bias {nullptr};
   cutlass::half_t* layernorm_14_0 {nullptr};
   cutlass::half_t* transformer_resblocks_0_mlp_c_fc_weight {nullptr};
   cutlass::half_t* transformer_resblocks_0_mlp_c_fc_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_15_0 {nullptr};
   cutlass::half_t* elementwise_18_0 {nullptr};
   cutlass::half_t* transformer_resblocks_0_mlp_c_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_0_mlp_c_proj_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_19_0 {nullptr};
   cutlass::half_t* transformer_resblocks_1_ln_1_weight {nullptr};
   cutlass::half_t* transformer_resblocks_1_ln_1_bias {nullptr};
   cutlass::half_t* layernorm_20_0 {nullptr};
   cutlass::half_t* elementwise_21_0 {nullptr};
   cutlass::half_t* transformer_resblocks_1_attn_qkv_weight {nullptr};
   cutlass::half_t* reshape_23_0 {nullptr};
   int32_t* transformer_resblocks_1_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_24_0 {nullptr};
   cutlass::half_t* transformer_resblocks_1_attn_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_1_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_27_0 {nullptr};
   cutlass::half_t* transformer_resblocks_1_ln_2_weight {nullptr};
   cutlass::half_t* transformer_resblocks_1_ln_2_bias {nullptr};
   cutlass::half_t* layernorm_28_0 {nullptr};
   cutlass::half_t* transformer_resblocks_1_mlp_c_fc_weight {nullptr};
   cutlass::half_t* transformer_resblocks_1_mlp_c_fc_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_29_0 {nullptr};
   cutlass::half_t* elementwise_32_0 {nullptr};
   cutlass::half_t* transformer_resblocks_1_mlp_c_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_1_mlp_c_proj_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_33_0 {nullptr};
   cutlass::half_t* transformer_resblocks_2_ln_1_weight {nullptr};
   cutlass::half_t* transformer_resblocks_2_ln_1_bias {nullptr};
   cutlass::half_t* layernorm_34_0 {nullptr};
   cutlass::half_t* elementwise_35_0 {nullptr};
   cutlass::half_t* transformer_resblocks_2_attn_qkv_weight {nullptr};
   cutlass::half_t* reshape_37_0 {nullptr};
   int32_t* transformer_resblocks_2_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_38_0 {nullptr};
   cutlass::half_t* transformer_resblocks_2_attn_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_2_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_41_0 {nullptr};
   cutlass::half_t* transformer_resblocks_2_ln_2_weight {nullptr};
   cutlass::half_t* transformer_resblocks_2_ln_2_bias {nullptr};
   cutlass::half_t* layernorm_42_0 {nullptr};
   cutlass::half_t* transformer_resblocks_2_mlp_c_fc_weight {nullptr};
   cutlass::half_t* transformer_resblocks_2_mlp_c_fc_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_43_0 {nullptr};
   cutlass::half_t* elementwise_46_0 {nullptr};
   cutlass::half_t* transformer_resblocks_2_mlp_c_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_2_mlp_c_proj_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_47_0 {nullptr};
   cutlass::half_t* transformer_resblocks_3_ln_1_weight {nullptr};
   cutlass::half_t* transformer_resblocks_3_ln_1_bias {nullptr};
   cutlass::half_t* layernorm_48_0 {nullptr};
   cutlass::half_t* elementwise_49_0 {nullptr};
   cutlass::half_t* transformer_resblocks_3_attn_qkv_weight {nullptr};
   cutlass::half_t* reshape_51_0 {nullptr};
   int32_t* transformer_resblocks_3_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_52_0 {nullptr};
   cutlass::half_t* transformer_resblocks_3_attn_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_3_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_55_0 {nullptr};
   cutlass::half_t* transformer_resblocks_3_ln_2_weight {nullptr};
   cutlass::half_t* transformer_resblocks_3_ln_2_bias {nullptr};
   cutlass::half_t* layernorm_56_0 {nullptr};
   cutlass::half_t* transformer_resblocks_3_mlp_c_fc_weight {nullptr};
   cutlass::half_t* transformer_resblocks_3_mlp_c_fc_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_57_0 {nullptr};
   cutlass::half_t* elementwise_60_0 {nullptr};
   cutlass::half_t* transformer_resblocks_3_mlp_c_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_3_mlp_c_proj_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_61_0 {nullptr};
   cutlass::half_t* transformer_resblocks_4_ln_1_weight {nullptr};
   cutlass::half_t* transformer_resblocks_4_ln_1_bias {nullptr};
   cutlass::half_t* layernorm_62_0 {nullptr};
   cutlass::half_t* elementwise_63_0 {nullptr};
   cutlass::half_t* transformer_resblocks_4_attn_qkv_weight {nullptr};
   cutlass::half_t* reshape_65_0 {nullptr};
   int32_t* transformer_resblocks_4_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_66_0 {nullptr};
   cutlass::half_t* transformer_resblocks_4_attn_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_4_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_69_0 {nullptr};
   cutlass::half_t* transformer_resblocks_4_ln_2_weight {nullptr};
   cutlass::half_t* transformer_resblocks_4_ln_2_bias {nullptr};
   cutlass::half_t* layernorm_70_0 {nullptr};
   cutlass::half_t* transformer_resblocks_4_mlp_c_fc_weight {nullptr};
   cutlass::half_t* transformer_resblocks_4_mlp_c_fc_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_71_0 {nullptr};
   cutlass::half_t* elementwise_74_0 {nullptr};
   cutlass::half_t* transformer_resblocks_4_mlp_c_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_4_mlp_c_proj_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_75_0 {nullptr};
   cutlass::half_t* transformer_resblocks_5_ln_1_weight {nullptr};
   cutlass::half_t* transformer_resblocks_5_ln_1_bias {nullptr};
   cutlass::half_t* layernorm_76_0 {nullptr};
   cutlass::half_t* elementwise_77_0 {nullptr};
   cutlass::half_t* transformer_resblocks_5_attn_qkv_weight {nullptr};
   cutlass::half_t* reshape_79_0 {nullptr};
   int32_t* transformer_resblocks_5_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_80_0 {nullptr};
   cutlass::half_t* transformer_resblocks_5_attn_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_5_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_83_0 {nullptr};
   cutlass::half_t* transformer_resblocks_5_ln_2_weight {nullptr};
   cutlass::half_t* transformer_resblocks_5_ln_2_bias {nullptr};
   cutlass::half_t* layernorm_84_0 {nullptr};
   cutlass::half_t* transformer_resblocks_5_mlp_c_fc_weight {nullptr};
   cutlass::half_t* transformer_resblocks_5_mlp_c_fc_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_85_0 {nullptr};
   cutlass::half_t* elementwise_88_0 {nullptr};
   cutlass::half_t* transformer_resblocks_5_mlp_c_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_5_mlp_c_proj_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_89_0 {nullptr};
   cutlass::half_t* transformer_resblocks_6_ln_1_weight {nullptr};
   cutlass::half_t* transformer_resblocks_6_ln_1_bias {nullptr};
   cutlass::half_t* layernorm_90_0 {nullptr};
   cutlass::half_t* elementwise_91_0 {nullptr};
   cutlass::half_t* transformer_resblocks_6_attn_qkv_weight {nullptr};
   cutlass::half_t* reshape_93_0 {nullptr};
   int32_t* transformer_resblocks_6_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_94_0 {nullptr};
   cutlass::half_t* transformer_resblocks_6_attn_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_6_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_97_0 {nullptr};
   cutlass::half_t* transformer_resblocks_6_ln_2_weight {nullptr};
   cutlass::half_t* transformer_resblocks_6_ln_2_bias {nullptr};
   cutlass::half_t* layernorm_98_0 {nullptr};
   cutlass::half_t* transformer_resblocks_6_mlp_c_fc_weight {nullptr};
   cutlass::half_t* transformer_resblocks_6_mlp_c_fc_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_99_0 {nullptr};
   cutlass::half_t* elementwise_102_0 {nullptr};
   cutlass::half_t* transformer_resblocks_6_mlp_c_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_6_mlp_c_proj_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_103_0 {nullptr};
   cutlass::half_t* transformer_resblocks_7_ln_1_weight {nullptr};
   cutlass::half_t* transformer_resblocks_7_ln_1_bias {nullptr};
   cutlass::half_t* layernorm_104_0 {nullptr};
   cutlass::half_t* elementwise_105_0 {nullptr};
   cutlass::half_t* transformer_resblocks_7_attn_qkv_weight {nullptr};
   cutlass::half_t* reshape_107_0 {nullptr};
   int32_t* transformer_resblocks_7_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_108_0 {nullptr};
   cutlass::half_t* transformer_resblocks_7_attn_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_7_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_111_0 {nullptr};
   cutlass::half_t* transformer_resblocks_7_ln_2_weight {nullptr};
   cutlass::half_t* transformer_resblocks_7_ln_2_bias {nullptr};
   cutlass::half_t* layernorm_112_0 {nullptr};
   cutlass::half_t* transformer_resblocks_7_mlp_c_fc_weight {nullptr};
   cutlass::half_t* transformer_resblocks_7_mlp_c_fc_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_113_0 {nullptr};
   cutlass::half_t* elementwise_116_0 {nullptr};
   cutlass::half_t* transformer_resblocks_7_mlp_c_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_7_mlp_c_proj_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_117_0 {nullptr};
   cutlass::half_t* transformer_resblocks_8_ln_1_weight {nullptr};
   cutlass::half_t* transformer_resblocks_8_ln_1_bias {nullptr};
   cutlass::half_t* layernorm_118_0 {nullptr};
   cutlass::half_t* elementwise_119_0 {nullptr};
   cutlass::half_t* transformer_resblocks_8_attn_qkv_weight {nullptr};
   cutlass::half_t* reshape_121_0 {nullptr};
   int32_t* transformer_resblocks_8_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_122_0 {nullptr};
   cutlass::half_t* transformer_resblocks_8_attn_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_8_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_125_0 {nullptr};
   cutlass::half_t* transformer_resblocks_8_ln_2_weight {nullptr};
   cutlass::half_t* transformer_resblocks_8_ln_2_bias {nullptr};
   cutlass::half_t* layernorm_126_0 {nullptr};
   cutlass::half_t* transformer_resblocks_8_mlp_c_fc_weight {nullptr};
   cutlass::half_t* transformer_resblocks_8_mlp_c_fc_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_127_0 {nullptr};
   cutlass::half_t* elementwise_130_0 {nullptr};
   cutlass::half_t* transformer_resblocks_8_mlp_c_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_8_mlp_c_proj_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_131_0 {nullptr};
   cutlass::half_t* transformer_resblocks_9_ln_1_weight {nullptr};
   cutlass::half_t* transformer_resblocks_9_ln_1_bias {nullptr};
   cutlass::half_t* layernorm_132_0 {nullptr};
   cutlass::half_t* elementwise_133_0 {nullptr};
   cutlass::half_t* transformer_resblocks_9_attn_qkv_weight {nullptr};
   cutlass::half_t* reshape_135_0 {nullptr};
   int32_t* transformer_resblocks_9_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_136_0 {nullptr};
   cutlass::half_t* transformer_resblocks_9_attn_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_9_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_139_0 {nullptr};
   cutlass::half_t* transformer_resblocks_9_ln_2_weight {nullptr};
   cutlass::half_t* transformer_resblocks_9_ln_2_bias {nullptr};
   cutlass::half_t* layernorm_140_0 {nullptr};
   cutlass::half_t* transformer_resblocks_9_mlp_c_fc_weight {nullptr};
   cutlass::half_t* transformer_resblocks_9_mlp_c_fc_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_141_0 {nullptr};
   cutlass::half_t* elementwise_144_0 {nullptr};
   cutlass::half_t* transformer_resblocks_9_mlp_c_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_9_mlp_c_proj_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_145_0 {nullptr};
   cutlass::half_t* transformer_resblocks_10_ln_1_weight {nullptr};
   cutlass::half_t* transformer_resblocks_10_ln_1_bias {nullptr};
   cutlass::half_t* layernorm_146_0 {nullptr};
   cutlass::half_t* elementwise_147_0 {nullptr};
   cutlass::half_t* transformer_resblocks_10_attn_qkv_weight {nullptr};
   cutlass::half_t* reshape_149_0 {nullptr};
   int32_t* transformer_resblocks_10_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_150_0 {nullptr};
   cutlass::half_t* transformer_resblocks_10_attn_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_10_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_153_0 {nullptr};
   cutlass::half_t* transformer_resblocks_10_ln_2_weight {nullptr};
   cutlass::half_t* transformer_resblocks_10_ln_2_bias {nullptr};
   cutlass::half_t* layernorm_154_0 {nullptr};
   cutlass::half_t* transformer_resblocks_10_mlp_c_fc_weight {nullptr};
   cutlass::half_t* transformer_resblocks_10_mlp_c_fc_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_155_0 {nullptr};
   cutlass::half_t* elementwise_158_0 {nullptr};
   cutlass::half_t* transformer_resblocks_10_mlp_c_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_10_mlp_c_proj_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_159_0 {nullptr};
   cutlass::half_t* transformer_resblocks_11_ln_1_weight {nullptr};
   cutlass::half_t* transformer_resblocks_11_ln_1_bias {nullptr};
   cutlass::half_t* layernorm_160_0 {nullptr};
   cutlass::half_t* elementwise_161_0 {nullptr};
   cutlass::half_t* transformer_resblocks_11_attn_qkv_weight {nullptr};
   cutlass::half_t* reshape_163_0 {nullptr};
   int32_t* transformer_resblocks_11_attn_cu_length {nullptr};
   cutlass::half_t* flash_attention_164_0 {nullptr};
   cutlass::half_t* transformer_resblocks_11_attn_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_11_attn_proj_bias {nullptr};
   cutlass::half_t* reshape_167_0 {nullptr};
   cutlass::half_t* transformer_resblocks_11_ln_2_weight {nullptr};
   cutlass::half_t* transformer_resblocks_11_ln_2_bias {nullptr};
   cutlass::half_t* layernorm_168_0 {nullptr};
   cutlass::half_t* transformer_resblocks_11_mlp_c_fc_weight {nullptr};
   cutlass::half_t* transformer_resblocks_11_mlp_c_fc_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_169_0 {nullptr};
   cutlass::half_t* elementwise_172_0 {nullptr};
   cutlass::half_t* transformer_resblocks_11_mlp_c_proj_weight {nullptr};
   cutlass::half_t* transformer_resblocks_11_mlp_c_proj_bias {nullptr};
   cutlass::half_t* gemm_rcr_bias_173_0 {nullptr};
   cutlass::half_t* permute102_174_0 {nullptr};
   cutlass::half_t* ln_final_weight {nullptr};
   cutlass::half_t* ln_final_bias {nullptr};
   cutlass::half_t* output_0 {nullptr};
   int64_t input_dim_0 { 1 };
   int64_t input_dim_1 { 77 };
   int64_t reshape_0_0_dim_0 { 77 };
   int64_t token_embedding_weight_dim_0 { 49408 };
   int64_t token_embedding_weight_dim_1 { 512 };
   int64_t batch_gather_1_0_dim_0 { 77 };
   int64_t batch_gather_1_0_dim_1 { 512 };
   int64_t positional_embedding_dim_0 { 77 };
   int64_t positional_embedding_dim_1 { 512 };
   int64_t reshape_3_0_dim_0 { 1 };
   int64_t reshape_3_0_dim_1 { 77 };
   int64_t reshape_3_0_dim_2 { 512 };
   int64_t transformer_resblocks_0_ln_1_weight_dim_0 { 512 };
   int64_t transformer_resblocks_0_ln_1_bias_dim_0 { 512 };
   int64_t transformer_resblocks_0_attn_qkv_weight_dim_0 { 1536 };
   int64_t transformer_resblocks_0_attn_qkv_weight_dim_1 { 512 };
   int64_t reshape_9_0_dim_0 { 77 };
   int64_t reshape_9_0_dim_1 { 3 };
   int64_t reshape_9_0_dim_2 { 8 };
   int64_t reshape_9_0_dim_3 { 64 };
   int64_t transformer_resblocks_0_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_10_0_dim_0 { 77 };
   int64_t flash_attention_10_0_dim_1 { 8 };
   int64_t flash_attention_10_0_dim_2 { 64 };
   int64_t transformer_resblocks_0_attn_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_0_attn_proj_weight_dim_1 { 512 };
   int64_t transformer_resblocks_0_attn_proj_bias_dim_0 { 512 };
   int64_t reshape_13_0_dim_0 { 77 };
   int64_t reshape_13_0_dim_1 { 1 };
   int64_t reshape_13_0_dim_2 { 512 };
   int64_t reshape_11_0_dim_0 { 77 };
   int64_t reshape_11_0_dim_1 { 512 };
   int64_t transformer_resblocks_0_ln_2_weight_dim_0 { 512 };
   int64_t transformer_resblocks_0_ln_2_bias_dim_0 { 512 };
   int64_t transformer_resblocks_0_mlp_c_fc_weight_dim_0 { 2048 };
   int64_t transformer_resblocks_0_mlp_c_fc_weight_dim_1 { 512 };
   int64_t transformer_resblocks_0_mlp_c_fc_bias_dim_0 { 2048 };
   int64_t transformer_resblocks_0_mlp_c_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_0_mlp_c_proj_weight_dim_1 { 2048 };
   int64_t transformer_resblocks_0_mlp_c_proj_bias_dim_0 { 512 };
   int64_t transformer_resblocks_1_ln_1_weight_dim_0 { 512 };
   int64_t transformer_resblocks_1_ln_1_bias_dim_0 { 512 };
   int64_t transformer_resblocks_1_attn_qkv_weight_dim_0 { 1536 };
   int64_t transformer_resblocks_1_attn_qkv_weight_dim_1 { 512 };
   int64_t reshape_23_0_dim_0 { 77 };
   int64_t reshape_23_0_dim_1 { 3 };
   int64_t reshape_23_0_dim_2 { 8 };
   int64_t reshape_23_0_dim_3 { 64 };
   int64_t transformer_resblocks_1_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_24_0_dim_0 { 77 };
   int64_t flash_attention_24_0_dim_1 { 8 };
   int64_t flash_attention_24_0_dim_2 { 64 };
   int64_t transformer_resblocks_1_attn_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_1_attn_proj_weight_dim_1 { 512 };
   int64_t transformer_resblocks_1_attn_proj_bias_dim_0 { 512 };
   int64_t reshape_27_0_dim_0 { 77 };
   int64_t reshape_27_0_dim_1 { 1 };
   int64_t reshape_27_0_dim_2 { 512 };
   int64_t reshape_25_0_dim_0 { 77 };
   int64_t reshape_25_0_dim_1 { 512 };
   int64_t transformer_resblocks_1_ln_2_weight_dim_0 { 512 };
   int64_t transformer_resblocks_1_ln_2_bias_dim_0 { 512 };
   int64_t transformer_resblocks_1_mlp_c_fc_weight_dim_0 { 2048 };
   int64_t transformer_resblocks_1_mlp_c_fc_weight_dim_1 { 512 };
   int64_t transformer_resblocks_1_mlp_c_fc_bias_dim_0 { 2048 };
   int64_t transformer_resblocks_1_mlp_c_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_1_mlp_c_proj_weight_dim_1 { 2048 };
   int64_t transformer_resblocks_1_mlp_c_proj_bias_dim_0 { 512 };
   int64_t transformer_resblocks_2_ln_1_weight_dim_0 { 512 };
   int64_t transformer_resblocks_2_ln_1_bias_dim_0 { 512 };
   int64_t transformer_resblocks_2_attn_qkv_weight_dim_0 { 1536 };
   int64_t transformer_resblocks_2_attn_qkv_weight_dim_1 { 512 };
   int64_t reshape_37_0_dim_0 { 77 };
   int64_t reshape_37_0_dim_1 { 3 };
   int64_t reshape_37_0_dim_2 { 8 };
   int64_t reshape_37_0_dim_3 { 64 };
   int64_t transformer_resblocks_2_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_38_0_dim_0 { 77 };
   int64_t flash_attention_38_0_dim_1 { 8 };
   int64_t flash_attention_38_0_dim_2 { 64 };
   int64_t transformer_resblocks_2_attn_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_2_attn_proj_weight_dim_1 { 512 };
   int64_t transformer_resblocks_2_attn_proj_bias_dim_0 { 512 };
   int64_t reshape_41_0_dim_0 { 77 };
   int64_t reshape_41_0_dim_1 { 1 };
   int64_t reshape_41_0_dim_2 { 512 };
   int64_t reshape_39_0_dim_0 { 77 };
   int64_t reshape_39_0_dim_1 { 512 };
   int64_t transformer_resblocks_2_ln_2_weight_dim_0 { 512 };
   int64_t transformer_resblocks_2_ln_2_bias_dim_0 { 512 };
   int64_t transformer_resblocks_2_mlp_c_fc_weight_dim_0 { 2048 };
   int64_t transformer_resblocks_2_mlp_c_fc_weight_dim_1 { 512 };
   int64_t transformer_resblocks_2_mlp_c_fc_bias_dim_0 { 2048 };
   int64_t transformer_resblocks_2_mlp_c_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_2_mlp_c_proj_weight_dim_1 { 2048 };
   int64_t transformer_resblocks_2_mlp_c_proj_bias_dim_0 { 512 };
   int64_t transformer_resblocks_3_ln_1_weight_dim_0 { 512 };
   int64_t transformer_resblocks_3_ln_1_bias_dim_0 { 512 };
   int64_t transformer_resblocks_3_attn_qkv_weight_dim_0 { 1536 };
   int64_t transformer_resblocks_3_attn_qkv_weight_dim_1 { 512 };
   int64_t reshape_51_0_dim_0 { 77 };
   int64_t reshape_51_0_dim_1 { 3 };
   int64_t reshape_51_0_dim_2 { 8 };
   int64_t reshape_51_0_dim_3 { 64 };
   int64_t transformer_resblocks_3_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_52_0_dim_0 { 77 };
   int64_t flash_attention_52_0_dim_1 { 8 };
   int64_t flash_attention_52_0_dim_2 { 64 };
   int64_t transformer_resblocks_3_attn_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_3_attn_proj_weight_dim_1 { 512 };
   int64_t transformer_resblocks_3_attn_proj_bias_dim_0 { 512 };
   int64_t reshape_55_0_dim_0 { 77 };
   int64_t reshape_55_0_dim_1 { 1 };
   int64_t reshape_55_0_dim_2 { 512 };
   int64_t reshape_53_0_dim_0 { 77 };
   int64_t reshape_53_0_dim_1 { 512 };
   int64_t transformer_resblocks_3_ln_2_weight_dim_0 { 512 };
   int64_t transformer_resblocks_3_ln_2_bias_dim_0 { 512 };
   int64_t transformer_resblocks_3_mlp_c_fc_weight_dim_0 { 2048 };
   int64_t transformer_resblocks_3_mlp_c_fc_weight_dim_1 { 512 };
   int64_t transformer_resblocks_3_mlp_c_fc_bias_dim_0 { 2048 };
   int64_t transformer_resblocks_3_mlp_c_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_3_mlp_c_proj_weight_dim_1 { 2048 };
   int64_t transformer_resblocks_3_mlp_c_proj_bias_dim_0 { 512 };
   int64_t transformer_resblocks_4_ln_1_weight_dim_0 { 512 };
   int64_t transformer_resblocks_4_ln_1_bias_dim_0 { 512 };
   int64_t transformer_resblocks_4_attn_qkv_weight_dim_0 { 1536 };
   int64_t transformer_resblocks_4_attn_qkv_weight_dim_1 { 512 };
   int64_t reshape_65_0_dim_0 { 77 };
   int64_t reshape_65_0_dim_1 { 3 };
   int64_t reshape_65_0_dim_2 { 8 };
   int64_t reshape_65_0_dim_3 { 64 };
   int64_t transformer_resblocks_4_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_66_0_dim_0 { 77 };
   int64_t flash_attention_66_0_dim_1 { 8 };
   int64_t flash_attention_66_0_dim_2 { 64 };
   int64_t transformer_resblocks_4_attn_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_4_attn_proj_weight_dim_1 { 512 };
   int64_t transformer_resblocks_4_attn_proj_bias_dim_0 { 512 };
   int64_t reshape_69_0_dim_0 { 77 };
   int64_t reshape_69_0_dim_1 { 1 };
   int64_t reshape_69_0_dim_2 { 512 };
   int64_t reshape_67_0_dim_0 { 77 };
   int64_t reshape_67_0_dim_1 { 512 };
   int64_t transformer_resblocks_4_ln_2_weight_dim_0 { 512 };
   int64_t transformer_resblocks_4_ln_2_bias_dim_0 { 512 };
   int64_t transformer_resblocks_4_mlp_c_fc_weight_dim_0 { 2048 };
   int64_t transformer_resblocks_4_mlp_c_fc_weight_dim_1 { 512 };
   int64_t transformer_resblocks_4_mlp_c_fc_bias_dim_0 { 2048 };
   int64_t transformer_resblocks_4_mlp_c_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_4_mlp_c_proj_weight_dim_1 { 2048 };
   int64_t transformer_resblocks_4_mlp_c_proj_bias_dim_0 { 512 };
   int64_t transformer_resblocks_5_ln_1_weight_dim_0 { 512 };
   int64_t transformer_resblocks_5_ln_1_bias_dim_0 { 512 };
   int64_t transformer_resblocks_5_attn_qkv_weight_dim_0 { 1536 };
   int64_t transformer_resblocks_5_attn_qkv_weight_dim_1 { 512 };
   int64_t reshape_79_0_dim_0 { 77 };
   int64_t reshape_79_0_dim_1 { 3 };
   int64_t reshape_79_0_dim_2 { 8 };
   int64_t reshape_79_0_dim_3 { 64 };
   int64_t transformer_resblocks_5_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_80_0_dim_0 { 77 };
   int64_t flash_attention_80_0_dim_1 { 8 };
   int64_t flash_attention_80_0_dim_2 { 64 };
   int64_t transformer_resblocks_5_attn_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_5_attn_proj_weight_dim_1 { 512 };
   int64_t transformer_resblocks_5_attn_proj_bias_dim_0 { 512 };
   int64_t reshape_83_0_dim_0 { 77 };
   int64_t reshape_83_0_dim_1 { 1 };
   int64_t reshape_83_0_dim_2 { 512 };
   int64_t reshape_81_0_dim_0 { 77 };
   int64_t reshape_81_0_dim_1 { 512 };
   int64_t transformer_resblocks_5_ln_2_weight_dim_0 { 512 };
   int64_t transformer_resblocks_5_ln_2_bias_dim_0 { 512 };
   int64_t transformer_resblocks_5_mlp_c_fc_weight_dim_0 { 2048 };
   int64_t transformer_resblocks_5_mlp_c_fc_weight_dim_1 { 512 };
   int64_t transformer_resblocks_5_mlp_c_fc_bias_dim_0 { 2048 };
   int64_t transformer_resblocks_5_mlp_c_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_5_mlp_c_proj_weight_dim_1 { 2048 };
   int64_t transformer_resblocks_5_mlp_c_proj_bias_dim_0 { 512 };
   int64_t transformer_resblocks_6_ln_1_weight_dim_0 { 512 };
   int64_t transformer_resblocks_6_ln_1_bias_dim_0 { 512 };
   int64_t transformer_resblocks_6_attn_qkv_weight_dim_0 { 1536 };
   int64_t transformer_resblocks_6_attn_qkv_weight_dim_1 { 512 };
   int64_t reshape_93_0_dim_0 { 77 };
   int64_t reshape_93_0_dim_1 { 3 };
   int64_t reshape_93_0_dim_2 { 8 };
   int64_t reshape_93_0_dim_3 { 64 };
   int64_t transformer_resblocks_6_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_94_0_dim_0 { 77 };
   int64_t flash_attention_94_0_dim_1 { 8 };
   int64_t flash_attention_94_0_dim_2 { 64 };
   int64_t transformer_resblocks_6_attn_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_6_attn_proj_weight_dim_1 { 512 };
   int64_t transformer_resblocks_6_attn_proj_bias_dim_0 { 512 };
   int64_t reshape_97_0_dim_0 { 77 };
   int64_t reshape_97_0_dim_1 { 1 };
   int64_t reshape_97_0_dim_2 { 512 };
   int64_t reshape_95_0_dim_0 { 77 };
   int64_t reshape_95_0_dim_1 { 512 };
   int64_t transformer_resblocks_6_ln_2_weight_dim_0 { 512 };
   int64_t transformer_resblocks_6_ln_2_bias_dim_0 { 512 };
   int64_t transformer_resblocks_6_mlp_c_fc_weight_dim_0 { 2048 };
   int64_t transformer_resblocks_6_mlp_c_fc_weight_dim_1 { 512 };
   int64_t transformer_resblocks_6_mlp_c_fc_bias_dim_0 { 2048 };
   int64_t transformer_resblocks_6_mlp_c_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_6_mlp_c_proj_weight_dim_1 { 2048 };
   int64_t transformer_resblocks_6_mlp_c_proj_bias_dim_0 { 512 };
   int64_t transformer_resblocks_7_ln_1_weight_dim_0 { 512 };
   int64_t transformer_resblocks_7_ln_1_bias_dim_0 { 512 };
   int64_t transformer_resblocks_7_attn_qkv_weight_dim_0 { 1536 };
   int64_t transformer_resblocks_7_attn_qkv_weight_dim_1 { 512 };
   int64_t reshape_107_0_dim_0 { 77 };
   int64_t reshape_107_0_dim_1 { 3 };
   int64_t reshape_107_0_dim_2 { 8 };
   int64_t reshape_107_0_dim_3 { 64 };
   int64_t transformer_resblocks_7_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_108_0_dim_0 { 77 };
   int64_t flash_attention_108_0_dim_1 { 8 };
   int64_t flash_attention_108_0_dim_2 { 64 };
   int64_t transformer_resblocks_7_attn_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_7_attn_proj_weight_dim_1 { 512 };
   int64_t transformer_resblocks_7_attn_proj_bias_dim_0 { 512 };
   int64_t reshape_111_0_dim_0 { 77 };
   int64_t reshape_111_0_dim_1 { 1 };
   int64_t reshape_111_0_dim_2 { 512 };
   int64_t reshape_109_0_dim_0 { 77 };
   int64_t reshape_109_0_dim_1 { 512 };
   int64_t transformer_resblocks_7_ln_2_weight_dim_0 { 512 };
   int64_t transformer_resblocks_7_ln_2_bias_dim_0 { 512 };
   int64_t transformer_resblocks_7_mlp_c_fc_weight_dim_0 { 2048 };
   int64_t transformer_resblocks_7_mlp_c_fc_weight_dim_1 { 512 };
   int64_t transformer_resblocks_7_mlp_c_fc_bias_dim_0 { 2048 };
   int64_t transformer_resblocks_7_mlp_c_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_7_mlp_c_proj_weight_dim_1 { 2048 };
   int64_t transformer_resblocks_7_mlp_c_proj_bias_dim_0 { 512 };
   int64_t transformer_resblocks_8_ln_1_weight_dim_0 { 512 };
   int64_t transformer_resblocks_8_ln_1_bias_dim_0 { 512 };
   int64_t transformer_resblocks_8_attn_qkv_weight_dim_0 { 1536 };
   int64_t transformer_resblocks_8_attn_qkv_weight_dim_1 { 512 };
   int64_t reshape_121_0_dim_0 { 77 };
   int64_t reshape_121_0_dim_1 { 3 };
   int64_t reshape_121_0_dim_2 { 8 };
   int64_t reshape_121_0_dim_3 { 64 };
   int64_t transformer_resblocks_8_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_122_0_dim_0 { 77 };
   int64_t flash_attention_122_0_dim_1 { 8 };
   int64_t flash_attention_122_0_dim_2 { 64 };
   int64_t transformer_resblocks_8_attn_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_8_attn_proj_weight_dim_1 { 512 };
   int64_t transformer_resblocks_8_attn_proj_bias_dim_0 { 512 };
   int64_t reshape_125_0_dim_0 { 77 };
   int64_t reshape_125_0_dim_1 { 1 };
   int64_t reshape_125_0_dim_2 { 512 };
   int64_t reshape_123_0_dim_0 { 77 };
   int64_t reshape_123_0_dim_1 { 512 };
   int64_t transformer_resblocks_8_ln_2_weight_dim_0 { 512 };
   int64_t transformer_resblocks_8_ln_2_bias_dim_0 { 512 };
   int64_t transformer_resblocks_8_mlp_c_fc_weight_dim_0 { 2048 };
   int64_t transformer_resblocks_8_mlp_c_fc_weight_dim_1 { 512 };
   int64_t transformer_resblocks_8_mlp_c_fc_bias_dim_0 { 2048 };
   int64_t transformer_resblocks_8_mlp_c_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_8_mlp_c_proj_weight_dim_1 { 2048 };
   int64_t transformer_resblocks_8_mlp_c_proj_bias_dim_0 { 512 };
   int64_t transformer_resblocks_9_ln_1_weight_dim_0 { 512 };
   int64_t transformer_resblocks_9_ln_1_bias_dim_0 { 512 };
   int64_t transformer_resblocks_9_attn_qkv_weight_dim_0 { 1536 };
   int64_t transformer_resblocks_9_attn_qkv_weight_dim_1 { 512 };
   int64_t reshape_135_0_dim_0 { 77 };
   int64_t reshape_135_0_dim_1 { 3 };
   int64_t reshape_135_0_dim_2 { 8 };
   int64_t reshape_135_0_dim_3 { 64 };
   int64_t transformer_resblocks_9_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_136_0_dim_0 { 77 };
   int64_t flash_attention_136_0_dim_1 { 8 };
   int64_t flash_attention_136_0_dim_2 { 64 };
   int64_t transformer_resblocks_9_attn_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_9_attn_proj_weight_dim_1 { 512 };
   int64_t transformer_resblocks_9_attn_proj_bias_dim_0 { 512 };
   int64_t reshape_139_0_dim_0 { 77 };
   int64_t reshape_139_0_dim_1 { 1 };
   int64_t reshape_139_0_dim_2 { 512 };
   int64_t reshape_137_0_dim_0 { 77 };
   int64_t reshape_137_0_dim_1 { 512 };
   int64_t transformer_resblocks_9_ln_2_weight_dim_0 { 512 };
   int64_t transformer_resblocks_9_ln_2_bias_dim_0 { 512 };
   int64_t transformer_resblocks_9_mlp_c_fc_weight_dim_0 { 2048 };
   int64_t transformer_resblocks_9_mlp_c_fc_weight_dim_1 { 512 };
   int64_t transformer_resblocks_9_mlp_c_fc_bias_dim_0 { 2048 };
   int64_t transformer_resblocks_9_mlp_c_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_9_mlp_c_proj_weight_dim_1 { 2048 };
   int64_t transformer_resblocks_9_mlp_c_proj_bias_dim_0 { 512 };
   int64_t transformer_resblocks_10_ln_1_weight_dim_0 { 512 };
   int64_t transformer_resblocks_10_ln_1_bias_dim_0 { 512 };
   int64_t transformer_resblocks_10_attn_qkv_weight_dim_0 { 1536 };
   int64_t transformer_resblocks_10_attn_qkv_weight_dim_1 { 512 };
   int64_t reshape_149_0_dim_0 { 77 };
   int64_t reshape_149_0_dim_1 { 3 };
   int64_t reshape_149_0_dim_2 { 8 };
   int64_t reshape_149_0_dim_3 { 64 };
   int64_t transformer_resblocks_10_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_150_0_dim_0 { 77 };
   int64_t flash_attention_150_0_dim_1 { 8 };
   int64_t flash_attention_150_0_dim_2 { 64 };
   int64_t transformer_resblocks_10_attn_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_10_attn_proj_weight_dim_1 { 512 };
   int64_t transformer_resblocks_10_attn_proj_bias_dim_0 { 512 };
   int64_t reshape_153_0_dim_0 { 77 };
   int64_t reshape_153_0_dim_1 { 1 };
   int64_t reshape_153_0_dim_2 { 512 };
   int64_t reshape_151_0_dim_0 { 77 };
   int64_t reshape_151_0_dim_1 { 512 };
   int64_t transformer_resblocks_10_ln_2_weight_dim_0 { 512 };
   int64_t transformer_resblocks_10_ln_2_bias_dim_0 { 512 };
   int64_t transformer_resblocks_10_mlp_c_fc_weight_dim_0 { 2048 };
   int64_t transformer_resblocks_10_mlp_c_fc_weight_dim_1 { 512 };
   int64_t transformer_resblocks_10_mlp_c_fc_bias_dim_0 { 2048 };
   int64_t transformer_resblocks_10_mlp_c_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_10_mlp_c_proj_weight_dim_1 { 2048 };
   int64_t transformer_resblocks_10_mlp_c_proj_bias_dim_0 { 512 };
   int64_t transformer_resblocks_11_ln_1_weight_dim_0 { 512 };
   int64_t transformer_resblocks_11_ln_1_bias_dim_0 { 512 };
   int64_t transformer_resblocks_11_attn_qkv_weight_dim_0 { 1536 };
   int64_t transformer_resblocks_11_attn_qkv_weight_dim_1 { 512 };
   int64_t reshape_163_0_dim_0 { 77 };
   int64_t reshape_163_0_dim_1 { 3 };
   int64_t reshape_163_0_dim_2 { 8 };
   int64_t reshape_163_0_dim_3 { 64 };
   int64_t transformer_resblocks_11_attn_cu_length_dim_0 { 2 };
   int64_t flash_attention_164_0_dim_0 { 77 };
   int64_t flash_attention_164_0_dim_1 { 8 };
   int64_t flash_attention_164_0_dim_2 { 64 };
   int64_t transformer_resblocks_11_attn_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_11_attn_proj_weight_dim_1 { 512 };
   int64_t transformer_resblocks_11_attn_proj_bias_dim_0 { 512 };
   int64_t reshape_167_0_dim_0 { 77 };
   int64_t reshape_167_0_dim_1 { 1 };
   int64_t reshape_167_0_dim_2 { 512 };
   int64_t reshape_165_0_dim_0 { 77 };
   int64_t reshape_165_0_dim_1 { 512 };
   int64_t transformer_resblocks_11_ln_2_weight_dim_0 { 512 };
   int64_t transformer_resblocks_11_ln_2_bias_dim_0 { 512 };
   int64_t transformer_resblocks_11_mlp_c_fc_weight_dim_0 { 2048 };
   int64_t transformer_resblocks_11_mlp_c_fc_weight_dim_1 { 512 };
   int64_t transformer_resblocks_11_mlp_c_fc_bias_dim_0 { 2048 };
   int64_t transformer_resblocks_11_mlp_c_proj_weight_dim_0 { 512 };
   int64_t transformer_resblocks_11_mlp_c_proj_weight_dim_1 { 2048 };
   int64_t transformer_resblocks_11_mlp_c_proj_bias_dim_0 { 512 };
   int64_t ln_final_weight_dim_0 { 512 };
   int64_t ln_final_bias_dim_0 { 512 };

};
} // namespace ait