
size_t GLOBAL_WORKSPACE_SIZE = 0;


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



  // Gemm operator cutlass_tensorop_h16816gemm_256x128_64x3_tn_align8
  using Operation_cutlass_tensorop_h16816gemm_256x128_64x3_tn_align8 = 
    cutlass::gemm::device::GemmUniversalWithBroadcast<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<256, 128, 64>,
            cutlass::gemm::GemmShape<64, 64, 64>,
            cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombinationResidualBlockV2<
            cutlass::half_t, cutlass::half_t, cutlass::half_t,
            cutlass::half_t,       8,
            cutlass::epilogue::thread::Identity, cutlass::plus, cutlass::epilogue::thread::Identity

        >,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
            3,
            8,
            8
    >;

using GemmInstance = Operation_cutlass_tensorop_h16816gemm_256x128_64x3_tn_align8;

void gemm (
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

  
//  TODO: cast to right dtype
  using ElementComputeEpilogue = typename GemmInstance::ElementAccumulator;

  typename GemmInstance::Arguments arguments{


    cutlass::gemm::GemmUniversalMode::kGemm,
    { M, N, K },

    split_k,

    {ElementComputeEpilogue(1), ElementComputeEpilogue(1)},
    (void*) a_ptr,
    (void*) b_ptr,
    (void*) d0_ptr,

    nullptr,

    (void*) (c_ptr + output_offset),
    (void*) bias_ptr,
    nullptr,
    /*batch_stride_A*/ 0,
    /*batch_stride_B*/ 0,
    /*batch_stride_C1*/ 0,
    /*batch_stride_C2*/ 0,
    /*batch_stride_D*/ 0,
    /*batch_stride_Vector*/ 0,
    /*batch_stride_Tensor*/ 0,
    K,
    K,
    N,

    0,

    output_stride,
    /*ldr*/ 0,
    /*/ldt*/ 0

  };
  GemmInstance gemm_op;

  // https://www.youtube.com/watch?v=rRwxfYlgG-M
  size_t workspace_size = gemm_op.get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> local_workspace(workspace_size);
  workspace = local_workspace.get();
  GLOBAL_WORKSPACE_SIZE = workspace_size;

  auto status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op.initialize(arguments, workspace, stream);
  CUTLASS_CHECK(status);
  status = gemm_op(stream);
  CUTLASS_CHECK(status);
  return;
  throw std::runtime_error(
      "Unsupported workload for this gemm specialization."
  );
}

struct ProfilerMemoryPool {
  ProfilerMemoryPool() {
    std::random_device rd;
    gen = std::mt19937(rd());
    uniform_dist = std::uniform_int_distribution<int64_t>(1, 48964896);
    offsets.reserve(512);
    strides.reserve(512);
    copies.reserve(512);
    ptrs.reserve(512);
    blobs.reserve(512);
  }
  ~ProfilerMemoryPool() {}

  template <typename DType>
  DType* AllocateGaussianTensor(int64_t size) {
    size_t length = size * sizeof(DType);
    blobs.emplace_back(length);
    DType* ptr = reinterpret_cast<DType*>(blobs.back().get());

    uint64_t seed = uniform_dist(gen);
    double mean = 0.f;
    double std = 1.f;

    cutlass::reference::device::BlockFillRandomGaussian(ptr, size, seed, mean,
                                                        std);

    return ptr;
  }


  cutlass::half_t* AllocateHalfGaussianTensor(int64_t size) {
    return reinterpret_cast<cutlass::half_t*>(
        AllocateGaussianTensor<__half>(size));
  }

  int AllocateHalfTensor(int64_t size, int64_t copy) {
    offsets.push_back(0);
    strides.push_back(size);
    copies.push_back(copy);
    auto ptr = AllocateHalfGaussianTensor(size * copy);
    ptrs.push_back(reinterpret_cast<void*>(ptr));
    return ptrs.size() - 1;
  }

  cutlass::half_t* RequestHalfTensorByIdx(int idx) {
    auto copy = copies.at(idx);
    auto offset = offsets.at(idx);
    auto stride = strides.at(idx);
    cutlass::half_t* ptr = reinterpret_cast<cutlass::half_t*>(ptrs.at(idx));
    ptr += offset;
    offset += stride;
    if (offset == copy * stride) {
        offset = 0;
    }
    offsets[idx] = offset;
    return ptr;
  }

  std::vector<int64_t> offsets;
  std::vector<int64_t> strides;
  std::vector<int64_t> copies;
  std::vector<void*> ptrs;
  std::vector<cutlass::DeviceAllocation<uint8_t> > blobs;
  std::mt19937 gen;
  std::uniform_int_distribution<int64_t> uniform_dist;
};


int main(int argc, char** argv) {
  int device_idx;
  cudaDeviceProp device_properties;
  cudaError_t result = cudaGetDevice(&device_idx);
  auto memory_pool = std::make_unique<ProfilerMemoryPool>();
  if (result != cudaSuccess) {
    throw std::runtime_error("cudaGetDevice() API call failed.");
  }

  result = cudaGetDeviceProperties(&device_properties, device_idx);

  if (result != cudaSuccess) {
    throw std::runtime_error("cudaGetDeviceProperties() failed");
  }

  
  int64_t M = std::atoi(argv[1]);
  int64_t N = std::atoi(argv[2]);
  int64_t K = std::atoi(argv[3]);

  int split_k = std::atoi(argv[4]);

  
  int64_t a_dim0 = M;
  int64_t a_dim1 = K;
  int64_t b_dim0 = N;
  int64_t b_dim1 = K;
  int64_t c_dim0 = M;
  int64_t c_dim1 = N;


  using ElementOutput = typename GemmInstance::ElementC;
  using ElementInputA = typename GemmInstance::ElementA;
  using ElementInputB = typename GemmInstance::ElementB;
  uint8_t* global_workspace = nullptr;
  cudaStream_t stream = nullptr;

  
  int64_t a_ptr_sz = a_dim0 * a_dim1;
  int64_t b_ptr_sz = b_dim0 * b_dim1;
  int64_t c_ptr_sz = c_dim0 * c_dim1;
  // The value 1 is used to force ptr_max_sz to be non-zero
  int64_t ptr_max_sz = std::max<int64_t>({1, a_ptr_sz, b_ptr_sz, c_ptr_sz});
  // TODO: special pool size for A100 L2 cache 40M
  // need to tune it for other devices
  int64_t mem_pool_sz = std::max(2,  std::min(64, int((1 << 25) / ptr_max_sz)));

  memory_pool->AllocateHalfTensor(a_ptr_sz, mem_pool_sz);  // a_ptr: index 0
  memory_pool->AllocateHalfTensor(b_ptr_sz, mem_pool_sz);  // b_ptr: index 1
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // c_ptr: index 2
  memory_pool->AllocateHalfTensor(c_dim1, mem_pool_sz);  // bias_ptr: index 3
  memory_pool->AllocateHalfTensor(c_ptr_sz, mem_pool_sz);  // d0 ptr: index 4


  // warmup
  for (int i = 0; i < 5; ++i) {
    
{

gemm(
    memory_pool->RequestHalfTensorByIdx(0),
    memory_pool->RequestHalfTensorByIdx(1),
    memory_pool->RequestHalfTensorByIdx(3),
    memory_pool->RequestHalfTensorByIdx(4),

    memory_pool->RequestHalfTensorByIdx(2),
    global_workspace,

 split_k,


    &a_dim0,

    &a_dim1,


    &b_dim0,

    &b_dim1,


    &c_dim0,

    &c_dim1,

    stream
);
}
  }
  cudaEvent_t events[2];
  for (auto & event : events) {
    cudaEventCreate(&event);
  }
  cudaEventRecord(events[0]);
  for (int i = 0; i < 10; ++i) {
    
{

gemm(
    memory_pool->RequestHalfTensorByIdx(0),
    memory_pool->RequestHalfTensorByIdx(1),
    memory_pool->RequestHalfTensorByIdx(3),
    memory_pool->RequestHalfTensorByIdx(4),

    memory_pool->RequestHalfTensorByIdx(2),
    global_workspace,

 split_k,


    &a_dim0,

    &a_dim1,


    &b_dim0,

    &b_dim1,


    &c_dim0,

    &c_dim1,

    stream
);
}
  }
  cudaEventRecord(events[1]);
  cudaEventSynchronize(events[1]);
  float runtime_ms = 0;
  cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }
  // TODO: output workspace
  if (runtime_ms < 0.00001) {
      throw std::runtime_error(
      "OOB in cutlass."
    );
  }
  std::cout << "TIME:" << runtime_ms << std::endl;
  std::cout << "WS:" << GLOBAL_WORKSPACE_SIZE << std::endl;
  return 0;
}