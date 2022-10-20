
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_residual_block_v2.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/device_memory.h"

#define CUTLASS_CHECK(status)                                                         \
  {                                                                                   \
    cutlass::Status error = status;                                                   \
    if (error != cutlass::Status::kSuccess) {                                         \
      auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +              \
          cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);         \
      std::cerr << msg << std::endl;                                                  \
      throw std::runtime_error(msg);                                                  \
    }                                                                                 \
  }



  // Gemm operator cutlass_tensorop_h16816gemm_64x64_64x5_tn_align8
  using Operation_cutlass_tensorop_h16816gemm_64x64_64x5_tn_align8 = 
    cutlass::gemm::device::GemmUniversalWithBroadcast<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<64, 64, 64>,
            cutlass::gemm::GemmShape<32, 32, 64>,
            cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombinationResidualBlockV2<
            cutlass::half_t, cutlass::half_t, cutlass::half_t,
            cutlass::half_t,       8,
            cutlass::epilogue::thread::Identity, cutlass::plus, cutlass::epilogue::thread::Identity

        >,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
            5,
            8,
            8
    >;

using f14f942549d5cf8281be76516a6eed92e602ed54f = Operation_cutlass_tensorop_h16816gemm_64x64_64x5_tn_align8;

void gemm_rcr_bias_add_12 (
    cutlass::half_t* a_ptr,
    cutlass::half_t* b_ptr,
    cutlass::half_t* bias_ptr,
    cutlass::half_t* d0_ptr,
    cutlass::half_t* c_ptr,
    uint8_t* workspace,
    int split_k,
    int64_t* a_dim0,
    int64_t* a_dim1,
    int64_t* b_dim0,
    int64_t* b_dim1,
    int64_t* c_dim0,
    int64_t* c_dim1,
    cudaStream_t stream
  ) {
  
 int64_t M = (*a_dim0);

 int64_t N = (*b_dim0);

 int64_t K = (*a_dim1);
  
  int64_t input_a_batch_stride = M * K;
  int64_t input_a_stride = K;
  int64_t input_a_offset = 0; // default to 0
  int64_t input_b_batch_stride = N * K;
  int64_t input_b_stride = K;
  int64_t input_b_offset = 0; // default to 0
    
  
  
  int64_t output_stride = N;
  int64_t output_offset = 0;
  
    
  
  
  int64_t a_size = 1;

    a_size *= *a_dim0;

    a_size *= *a_dim1;

  if (a_size != 0 && !a_ptr) {
    throw std::runtime_error("input a is null!");
  }

  int64_t b_size = 1;

    b_size *= *b_dim0;

    b_size *= *b_dim1;

  if (b_size != 0 && !b_ptr) {
    throw std::runtime_error("input b is null!");
  }

  int64_t c_size = 1;

    c_size *= *c_dim0;

    c_size *= *c_dim1;

  if (c_size != 0) {
    if (!c_ptr) {
      throw std::runtime_error("input c is null!");
    }
  } else {
    // output is empty and safe to return
    return;
  }

  // One of the input tensor are empty
  if (a_size == 0 || b_size == 0) {
    return;
  }

  if (!bias_ptr) {
    throw std::runtime_error("bias is null!");
  }
  if (!d0_ptr) {
    throw std::runtime_error("d0_ptr is null!");
  }

  
  if (M == 77 && N == 512 && K == 512) {
    
//  TODO: cast to right dtype
    using ElementComputeEpilogue = typename f14f942549d5cf8281be76516a6eed92e602ed54f::ElementAccumulator;

    typename f14f942549d5cf8281be76516a6eed92e602ed54f::Arguments arguments{


    cutlass::gemm::GemmUniversalMode::kGemm,
    { M, N, K },

    split_k,

    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},
    (void*) (a_ptr + input_a_offset),
    (void*) (b_ptr + input_b_offset),
    (void*) d0_ptr,

    nullptr,

    (void*) (c_ptr + output_offset),
    (void*) bias_ptr,
    nullptr,
    /*batch_stride_A*/ input_a_batch_stride,
    /*batch_stride_B*/ input_b_batch_stride,
    /*batch_stride_C1*/ 0,
    /*batch_stride_C2*/ 0,
    /*batch_stride_D*/ 0,
    /*batch_stride_Vector*/ 0,
    /*batch_stride_Tensor*/ 0,
    input_a_stride,
    input_b_stride,
    N,

    0,

    output_stride,
    /*ldr*/ 0,
    /*/ldt*/ 0

    };
    f14f942549d5cf8281be76516a6eed92e602ed54f gemm_op;

    auto status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
    status = gemm_op.initialize(arguments, workspace, stream);
    CUTLASS_CHECK(status);
    status = gemm_op(stream);
    CUTLASS_CHECK(status);
    return;
  }
  throw std::runtime_error(
      "Unsupported workload for this gemm_rcr_bias_add_12 specialization."
  );
}