
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"

#include "fmha.h"
#include "fmha_fprop_kernel_1xN.h"

namespace {

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Return_softmax>
__global__ void fmha_fprop_fp16_sm80_loop_kernel(Fused_multihead_attention_fprop_params params) {
    fmha::device_1xN_loop<Kernel_traits, Is_dropout, Is_causal, Return_softmax>(params);
}

template<typename Kernel_traits>
void run_fmha_fp16_sm80_loop_(Launch_params<Fused_multihead_attention_fprop_params> &launch_params,
                            const bool configure) {
    bool is_causal = launch_params.params.is_causal;
    auto kernel = (is_causal
           ? (&fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, false, true, false>)
           : (&fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, false, false, false>));

    constexpr int N = Kernel_traits::Cta_tile_p::N;
    const int loop_steps = (launch_params.params.s + N - 1) / N;
    constexpr int smem_size_softmax_lse = Kernel_traits::Smem_dp_sum::BYTES_PER_TILE;
    // Don't need smem_size_softmax_lse if we're not looping
    const int smem_size = fmha::get_dynamic_smem_size<Kernel_traits>()
        + (loop_steps > 1 ? smem_size_softmax_lse : 0);

    if( smem_size >= 48 * 1024 ) {
        FMHA_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    if (configure) {
        using Mma_tile_p = fmha::Hmma_tile<typename Kernel_traits::Cta_tile_p>;
        constexpr int M = Kernel_traits::Cta_tile_p::M;
        size_t STEPS = (launch_params.params.s + M - 1) / M;
        constexpr size_t MMAS_M = Mma_tile_p::MMAS_M;
        constexpr size_t MMAS_N = Mma_tile_p::MMAS_N;
        size_t elts_per_head = STEPS * MMAS_M * MMAS_N * 8 * loop_steps;
        launch_params.elts_per_thread = elts_per_head;
        return;
    }

    dim3 grid(launch_params.params.h, launch_params.params.b);
    kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(
        launch_params.params);

    FMHA_CHECK_CUDA(cudaPeekAtLastError());
}

void run_fmha_fp16_sm80(Launch_params<Fused_multihead_attention_fprop_params> &launch_params,
                        const bool configure) {

    using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 4, 0x08u>;
    run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
    
}

void set_params(Fused_multihead_attention_fprop_params &params,
                // sizes
                const size_t b,
                const size_t s,
                const size_t h,
                const size_t d,
                // device pointers
                void *qkv_packed_d,
                void *cu_seqlens_d,
                void *o_packed_d,
                void *o_tmp_d,
                void *do_packed_d,
                void *s_d,
                void *softmax_lse_d,
                void *dsoftmax_sum_d,
                float p_dropout,
                float softmax_scale,
                bool is_causal) {

    Data_type acc_type = DATA_TYPE_FP32;
    Data_type data_type = DATA_TYPE_FP16;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = qkv_packed_d;
    params.k_ptr = qkv_packed_d + get_size_in_bytes(h * d, data_type);
    params.v_ptr = qkv_packed_d + 2 * get_size_in_bytes(h * d, data_type);
    params.q_row_stride_in_elts = 3 * h * d;
    params.k_row_stride_in_elts = 3 * h * d;
    params.v_row_stride_in_elts = 3 * h * d;
    params.q_head_stride_in_elts = d;
    params.k_head_stride_in_elts = d;
    params.v_head_stride_in_elts = d;
    params.o_ptr = o_packed_d;
    params.o_row_stride_in_elts = h * d;
    params.o_head_stride_in_elts = d;
    params.do_ptr = do_packed_d;
    params.o_tmp_ptr = o_tmp_d;

    params.cu_seqlens = static_cast<int *>(cu_seqlens_d);

    // S = softmax(P)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * s, data_type);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;
    params.dsoftmax_sum = dsoftmax_sum_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.s = s;
    params.d = d;

    // Set the different scale values.
    // const float scale_bmm1 = 1.f / sqrtf(d);
    const float scale_bmm1 = softmax_scale;
    constexpr float scale_softmax = 1.f;
    constexpr float scale_bmm2 = 1.f;

    params.scale_bmm1f = scale_bmm1;
    set_alpha(params.scale_bmm1, scale_bmm1, data_type);
    set_alpha(params.scale_softmax, scale_softmax, acc_type);
    set_alpha(params.scale_bmm2, scale_bmm2, data_type);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.rp_dropout = 1.f / params.p_dropout;
    set_alpha(params.scale_dropout, params.rp_dropout, data_type);

    params.is_causal = is_causal;
}
}  // namespace


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
                   cudaStream_t stream)
    
{
    bool is_dropout = p_dropout > 0.0;
    bool return_softmax = false;

    Launch_params<Fused_multihead_attention_fprop_params> launch_params(stream, is_dropout, return_softmax);

    set_params(launch_params.params,
               batch_size, // b
               seq_len, // s
               num_heads, // h
               head_size, // d
               (void*)qkv,
               (void*)cu_seqlens,
               (void*)output,
               loop ? (void*)o_tmp : nullptr,
               nullptr,
               nullptr, // return softmax
               (void*)softmax_lse,
               nullptr,
               p_dropout,
               softmax_scale,
               is_causal);

    run_fmha_fp16_sm80(launch_params, /*configure=*/ false);
}
    