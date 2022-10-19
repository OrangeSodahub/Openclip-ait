

#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"


namespace {


const int64_t kThreadsNumPerBlock = 256;
const int64_t kMaxBlocksNum = 8192;

#define GPU_KERNEL_LOOP(i, n)                                   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);        i += blockDim.x * gridDim.x)

template <typename K>
__device__ int64_t GetInOffset(
    const int64_t out_offset,
    const K* indices,
    const int64_t indices_num,
    const int64_t instance_size,
    const int64_t gather_dim_size) {
  const int64_t batch_idx = out_offset / (indices_num * instance_size);
  const int64_t indices_idx =
      out_offset % (indices_num * instance_size) / instance_size;
  const int64_t inner_idx = out_offset % instance_size;
  const int64_t idx = indices[batch_idx * indices_num + indices_idx];
  assert(idx >= 0 && idx < gather_dim_size);
  return batch_idx * gather_dim_size * instance_size + idx * instance_size +
      inner_idx;
}

template <typename T, typename K>
__global__ void BatchGatherGpu(
    const int64_t elem_cnt,
    const T* in,
    const K* indices,
    const int64_t indices_num,
    const int64_t instance_size,
    const int64_t gather_dim_size,
    T* out) {
  GPU_KERNEL_LOOP(i, elem_cnt) {
    out[i] = in[GetInOffset<K>(
        i, indices, indices_num, instance_size, gather_dim_size)];
  }
}

inline int64_t BlocksNum4ThreadsNum(const int64_t n) {
  return std::min(
      (n + kThreadsNumPerBlock - 1) / kThreadsNumPerBlock,
      kMaxBlocksNum);
}
template <typename T, typename K>
void batch_gather_launcher(
    cudaStream_t stream,
    const int64_t batch_num,
    const int64_t indices_num,
    const int64_t instance_size,
    const int64_t gather_dim_size,
    const T* input,
    const K* indices,
    void* workspace,
    T* output) {
  const int64_t elem_cnt = batch_num * indices_num * instance_size;
  BatchGatherGpu<T, K>
      <<<BlocksNum4ThreadsNum(elem_cnt), kThreadsNumPerBlock, 0, stream>>>(
          elem_cnt,
          input,
          indices,
          indices_num,
          instance_size,
          gather_dim_size,
          output);
}
    

}  // namespace


void batch_gather_1(half* output,
                   const half* input,
                   const int64_t* indices,
                   const int64_t batch_num,
                   const int64_t indices_num,
                   const int64_t instance_size,
                   const int64_t gather_dim_size,
                   uint8_t* workspace,
                   cudaStream_t stream)
    
{
    batch_gather_launcher<half, int64_t>(stream, batch_num, indices_num, instance_size, gather_dim_size, input, indices, workspace, output);
}
    