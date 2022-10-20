

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/util/host_tensor.h"


#define TILE_SIZE 32
#define CH_K 4

namespace {
template <typename T>
__global__ void nhwc_to_nchw_kernel(T *output,
                                    const T *input,
                                    const int n,
                                    const int h,
                                    const int w,
                                    const int c) {

  const int hw = h*w;
  const int hwc = hw*c;
  __shared__ T shbuf[TILE_SIZE * (TILE_SIZE + 1)];
  const int32_t tid  = threadIdx.y*blockDim.x + threadIdx.x;
  const int32_t wid  = tid / TILE_SIZE;//th.y:0-7
  const int32_t lid  = tid % TILE_SIZE;//th.x:0-31
  const int32_t ni0   = blockIdx.z;
  const int32_t hwi0  = blockIdx.y * TILE_SIZE;//parallel 8*seq 4
  const int32_t ci0 = blockIdx.x * TILE_SIZE;//parallel 32
  const size_t input_idx = ni0 * hwc + (hwi0 + wid) * c + ci0;
  const T *A = input + input_idx;
  if (ci0 + lid < c) {
    const int lid_x_33 = lid * (TILE_SIZE + 1);
    if ((hwi0 + TILE_SIZE - TILE_SIZE / CH_K) <= hw) {
      int hwi = wid;  // between 0 and 7
      #pragma unroll
      for (int cLoopIdx = 0; cLoopIdx < CH_K; cLoopIdx++) {
        shbuf[lid_x_33 + hwi] = A[lid];
        A                     = &A[TILE_SIZE / CH_K * c];//because c is distributed on threads y
        hwi += TILE_SIZE / CH_K;
      }
    } else {
      for (int hwi = wid; hwi < TILE_SIZE; hwi += TILE_SIZE / CH_K) {
        if ((hwi + hwi0) < hw) {
          shbuf[lid_x_33 + hwi] = A[lid];
        }
        A = &A[TILE_SIZE / CH_K * c];
      }
    }
  }
  __syncthreads();

  const int32_t hwiOut = hwi0 + lid;
  const int nc = n*c;
  output = &output[hwiOut*nc];
  if(hwiOut < hw){
    if(ci0 + TILE_SIZE < c){
      int cI = wid;
      #pragma unroll
      for(int hwLoopIdx = 0; hwLoopIdx < CH_K; ++hwLoopIdx){
          output[ni0*c + ci0 + cI] = shbuf[(cI)* (TILE_SIZE + 1) + lid];
          cI += TILE_SIZE / CH_K;
      }
    } else {
      for(int cI = wid; cI < TILE_SIZE; cI += TILE_SIZE / CH_K){
        if(ci0+cI<c){
          output[ni0*c+ci0+cI] = shbuf[(cI)* (TILE_SIZE + 1) + lid];
        }
      }
    }
  }
}

void permute102_launcher(cutlass::half_t* in_ptr,
                         cutlass::half_t* out_ptr,
                         int x_dim0,
                         int x_dim1,
                         int x_dim2,
                         cudaStream_t stream) {
  const int n = x_dim0;
  const int h = 1;
  const int w = x_dim1;
  const int c = x_dim2;
  dim3 grid((c + TILE_SIZE - 1)/TILE_SIZE, (h*w + TILE_SIZE -1)/TILE_SIZE, n);
  dim3 block(TILE_SIZE, TILE_SIZE / CH_K);
  nhwc_to_nchw_kernel<cutlass::half_t><<<grid, block, 0, stream>>>(
    out_ptr,
    (const cutlass::half_t*)in_ptr,
    n,
    h,
    w,
    c
  );
}
} // namespace

void permute102_5 (
    cutlass::half_t* in_ptr,
    cutlass::half_t* out_ptr,
    int64_t* x_dim0,
    int64_t* x_dim1,
    int64_t* x_dim2,
    int64_t* y_dim0,
    int64_t* y_dim1,
    int64_t* y_dim2,
    cudaStream_t stream
) {
  if (!in_ptr) {
    throw std::runtime_error("in_ptr is NULL!");
  }
  if (!out_ptr) {
    throw std::runtime_error("in_ptr is NULL!");
  }
  
  int64_t X_DIM0 = *x_dim0;
  int64_t X_DIM1 = *x_dim1;
  int64_t X_DIM2 = *x_dim2;
  int64_t Y_DIM0 = X_DIM1;
  int64_t Y_DIM1 = X_DIM0;
  int64_t Y_DIM2 = X_DIM2;
  *y_dim0 = Y_DIM0;
  *y_dim1 = Y_DIM1;
  *y_dim2 = Y_DIM2;
  
permute102_launcher(
    in_ptr,
    out_ptr,
    *x_dim0,
    *x_dim1,
    *x_dim2,
    stream
);
return;
}
