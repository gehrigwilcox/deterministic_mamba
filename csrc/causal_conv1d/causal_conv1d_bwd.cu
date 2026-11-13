/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 *
 * Deterministic fork (based on causal-conv1d 1.5.0.post8, as used by mamba_ssm 2.2.4).
 * Upstream accumulates dweight / dbias across the batch with atomicAdd. Here each
 * (batch, channel) block stores its partial into a float32 workspace of shape
 * (batch, dim, width) / (batch, dim), which the host reduces in a fixed order.
 * The channel-last kernel (only used by Mamba2 / seq_idx) is removed.
 ******************************************************************************/

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#ifndef USE_ROCM
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #include <cub/block/block_reduce.cuh>
#else
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
#endif

#include "causal_conv1d.h"
#include "causal_conv1d_common.h"
#include "static_switch.h"

template<int kNThreads_, int kWidth_, bool kSiluAct_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct Causal_conv1d_bwd_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr bool kSiluAct = kSiluAct_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    static_assert(kWidth <= kNElts);
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangeRounds = sizeof(float) / sizeof(input_t);
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNElts, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, 1, cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNElts, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, 1, cub::BLOCK_STORE_DIRECT>;
    using BlockReduceFloatT = cub::BlockReduce<float, kNThreads>;
    static constexpr int kSmemIOSize = kIsVecLoad
        ? 0
        : custom_max({sizeof(typename BlockLoadT::TempStorage), sizeof(typename BlockStoreT::TempStorage)});
    static constexpr int kSmemExchangeSize = kNThreads * kNBytes * kNElts * (!kSiluAct ? 1 : kNExchangeRounds + 1);
    static constexpr int kSmemSize = custom_max({kSmemExchangeSize,
            int(sizeof(typename BlockReduceFloatT::TempStorage))}) + (kIsVecLoad ? 0 : kSmemIOSize);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_bwd_kernel(ConvParamsBwd params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr bool kSiluAct = Ktraits::kSiluAct;
    static constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNExchangeRounds = Ktraits::kNExchangeRounds;
    static constexpr bool kIsVecLoad = Ktraits::kIsVecLoad;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_);
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_);
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_ + Ktraits::kSmemIOSize);
    vec_t *smem_exchange_x = reinterpret_cast<vec_t *>(smem_ + Ktraits::kSmemIOSize) + kNThreads * kNExchangeRounds;
    auto& smem_reduce_float = *reinterpret_cast<typename Ktraits::BlockReduceFloatT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + dim_id * params.x_c_stride;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + dim_id * params.weight_c_stride;
    input_t *dout = reinterpret_cast<input_t *>(params.dout_ptr) + batch_id * params.dout_batch_stride
        + dim_id * params.dout_c_stride;
    input_t *dx = reinterpret_cast<input_t *>(params.dx_ptr) + batch_id * params.dx_batch_stride
        + dim_id * params.dx_c_stride;
    // Per-(batch, channel) workspace slots: dweight_ws (batch, dim, width), dbias_ws (batch, dim).
    const size_t bd_idx = size_t(batch_id) * params.dim + dim_id;
    float *dweight = reinterpret_cast<float *>(params.dweight_ptr) + bd_idx * kWidth;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[dim_id]);

    // Thread kNThreads - 1 will load the first elements of the next chunk so we initialize those to 0.
    if (tidx == 0) {
        if constexpr (!kSiluAct) {
            input_t zeros[kNElts] = {0};
            smem_exchange[0] = reinterpret_cast<vec_t *>(zeros)[0];
        } else {
            float zeros[kNElts] = {0};
            #pragma unroll
            for (int r = 0; r < kNExchangeRounds; ++r) {
                smem_exchange[r * kNThreads] = reinterpret_cast<vec_t *>(zeros)[r];
            }
        }
    }

    float weight_vals[kWidth];
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { weight_vals[i] = weight[i * params.weight_width_stride]; }

    float dweight_vals[kWidth] = {0};
    float dbias_val = 0;

    constexpr int kChunkSize = kNThreads * kNElts;
    const int n_chunks = (params.seqlen + kChunkSize - 1) / kChunkSize;
    x += (n_chunks - 1) * kChunkSize;
    dout += (n_chunks - 1) * kChunkSize;
    dx += (n_chunks - 1) * kChunkSize;
    for (int chunk = n_chunks - 1; chunk >= 0; --chunk) {
        input_t x_vals_load[2 * kNElts] = {0};
        input_t dout_vals_load[2 * kNElts] = {0};
        if constexpr(kIsVecLoad) {
            typename Ktraits::BlockLoadVecT(smem_load_vec).Load(reinterpret_cast<vec_t*>(x), *reinterpret_cast<vec_t (*)[1]>(&x_vals_load[kNElts]), (params.seqlen - chunk * kChunkSize) / kNElts);
            typename Ktraits::BlockLoadVecT(smem_load_vec).Load(reinterpret_cast<vec_t*>(dout), *reinterpret_cast<vec_t (*)[1]>(&dout_vals_load[0]), (params.seqlen - chunk * kChunkSize) / kNElts);
        } else {
            __syncthreads();
            typename Ktraits::BlockLoadT(smem_load).Load(x, *reinterpret_cast<input_t (*)[kNElts]>(&x_vals_load[kNElts]), params.seqlen - chunk * kChunkSize);
            __syncthreads();
            typename Ktraits::BlockLoadT(smem_load).Load(dout, *reinterpret_cast<input_t (*)[kNElts]>(&dout_vals_load[0]), params.seqlen - chunk * kChunkSize);
        }
        float dout_vals[2 * kNElts], x_vals[2 * kNElts];
        if constexpr (!kSiluAct) {
            __syncthreads();
            // Thread 0 don't write yet, so that thread kNThreads - 1 can read
            // the first elements of the next chunk.
            if (tidx > 0) { smem_exchange[tidx] = reinterpret_cast<vec_t *>(dout_vals_load)[0]; }
            __syncthreads();
            reinterpret_cast<vec_t *>(dout_vals_load)[1] = smem_exchange[tidx < kNThreads - 1 ? tidx + 1 : 0];
            __syncthreads();
            // Now thread 0 can write the first elements of the current chunk.
            if (tidx == 0) { smem_exchange[tidx] = reinterpret_cast<vec_t *>(dout_vals_load)[0]; }
            #pragma unroll
            for (int i = 0; i < 2 * kNElts; ++i) {
                dout_vals[i] = float(dout_vals_load[i]);
                x_vals[i] = float(x_vals_load[i]);
            }
        } else {
            if (tidx == 0 && chunk > 0) {
                if constexpr(kIsVecLoad) {
                    reinterpret_cast<vec_t *>(x_vals_load)[0] = reinterpret_cast<vec_t *>(x)[-1];
                } else {
                    #pragma unroll
                    for (int i = 0; i < kNElts; ++i) {
                        // Bug fix vs upstream causal-conv1d 1.5.0.post8: these are the last kNElts
                        // elements of the *previous* chunk (always in range when chunk > 0). Upstream
                        // bounds-checked the current chunk's index instead, which left them zero when
                        // the final chunk has < kNElts valid elements (e.g. seqlen = 2049), corrupting
                        // the recomputed SiLU input and hence dx / dweight / dbias at that boundary.
                        if (chunk * kChunkSize - kNElts + i < params.seqlen) { x_vals_load[i] = x[-kNElts + i]; }
                    }
                }
            }
            __syncthreads();
            smem_exchange_x[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[1];
            __syncthreads();
            if (tidx > 0) { reinterpret_cast<vec_t *>(x_vals_load)[0] = smem_exchange_x[tidx - 1]; }
            #pragma unroll
            for (int i = 0; i < 2 * kNElts; ++i) { x_vals[i] = float(x_vals_load[i]); }
            // Recompute the output
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                float out_val = bias_val;
                #pragma unroll
                for (int w = 0; w < kWidth; ++w) {
                    out_val += weight_vals[w] * x_vals[kNElts + i - (kWidth - w - 1)];
                }
                float out_sigmoid_val = 1.0f / (1.0f + expf(-out_val));
                dout_vals[i] = float(dout_vals_load[i]) * out_sigmoid_val
                               * (1.0f + out_val * (1.0f - out_sigmoid_val));
            }
            // Exchange the dout_vals. It's possible that we need to do 2 rounds of exchange
            // if input_t is 16 bits (since then we'd have 8 values of float)
            __syncthreads();
            // Thread 0 don't write yet, so that thread kNThreads - 1 can read
            // the first elements of the next chunk.
            if (tidx > 0) {
                #pragma unroll
                for (int r = 0; r < kNExchangeRounds; ++r) {
                    smem_exchange[r * kNThreads + tidx] = reinterpret_cast<vec_t *>(dout_vals)[r];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int r = 0; r < kNExchangeRounds; ++r) {
                reinterpret_cast<vec_t *>(dout_vals)[kNExchangeRounds + r]
                    = smem_exchange[r * kNThreads + (tidx < kNThreads - 1 ? tidx + 1 : 0)];
            }
            __syncthreads();
            // Now thread 0 can write the first elements of the current chunk.
            if (tidx == 0) {
                #pragma unroll
                for (int r = 0; r < kNExchangeRounds; ++r) {
                    smem_exchange[r * kNThreads + tidx] = reinterpret_cast<vec_t *>(dout_vals)[r];
                }
            }
        }
        dout -= kChunkSize;
        x -= kChunkSize;

        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { dbias_val += dout_vals[i]; }

        float dx_vals[kNElts] = {0};
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                dx_vals[i] += weight_vals[w] * dout_vals[i + kWidth - w - 1];
            }
        }

        input_t dx_vals_store[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { dx_vals_store[i] = dx_vals[i]; }
        if constexpr(kIsVecLoad) {
            typename Ktraits::BlockStoreVecT(smem_store_vec).Store(reinterpret_cast<vec_t*>(dx), reinterpret_cast<vec_t (&)[1]>(dx_vals_store), (params.seqlen - chunk * kChunkSize) / kNElts);
        } else {
            typename Ktraits::BlockStoreT(smem_store).Store(dx, dx_vals_store, params.seqlen - chunk * kChunkSize);
        }
        dx -= kChunkSize;

        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                dweight_vals[w] += x_vals[kNElts + i] * dout_vals[i + kWidth - w - 1];
            }
        }
    }

    #pragma unroll
    for (int w = 0; w < kWidth; ++w) {
        __syncthreads();
        dweight_vals[w] = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dweight_vals[w]);
        if (tidx == 0) {
            dweight[w] = dweight_vals[w];
        }
    }
    if (params.bias_ptr != nullptr) {
        __syncthreads();
        dbias_val = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dbias_val);
        if (tidx == 0) {
            reinterpret_cast<float *>(params.dbias_ptr)[bd_idx] = dbias_val;
        }
    }
}

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_bwd_launch(ConvParamsBwd &params, cudaStream_t stream) {
    static constexpr int kNElts = sizeof(input_t) == 4 ? 4 : 8;
    BOOL_SWITCH(params.seqlen % kNElts == 0, kIsVecLoad, [&] {
        BOOL_SWITCH(params.silu_activation, kSiluAct, [&] {
            using Ktraits = Causal_conv1d_bwd_kernel_traits<kNThreads, kWidth, kSiluAct, kIsVecLoad, input_t, weight_t>;
            constexpr int kSmemSize = Ktraits::kSmemSize;
            dim3 grid(params.batch, params.dim);
            auto kernel = &causal_conv1d_bwd_kernel<Ktraits>;

            if (kSmemSize >= 48 * 1024) {
                #ifndef USE_ROCM
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                #else
                // There is a slight signature discrepancy in HIP and CUDA "FuncSetAttribute" function.
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    (void *) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                std::cerr << "Warning (causal_conv1d bwd launch): attempting to set maxDynamicSharedMemorySize on an AMD GPU which is currently a non-op (in ROCm versions <= 6.1). This might lead to undefined behavior. \n" << std::endl;
                #endif
            }


            kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

template<typename input_t, typename weight_t>
void causal_conv1d_bwd_cuda(ConvParamsBwd &params, cudaStream_t stream) {
    if (params.width == 2) {
        causal_conv1d_bwd_launch<128, 2, input_t, weight_t>(params, stream);
    } else if (params.width == 3) {
        causal_conv1d_bwd_launch<128, 3, input_t, weight_t>(params, stream);
    } else if (params.width == 4) {
        causal_conv1d_bwd_launch<128, 4, input_t, weight_t>(params, stream);
    }
}

template void causal_conv1d_bwd_cuda<float, float>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_bwd_cuda<at::Half, float>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_bwd_cuda<at::BFloat16, float>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_bwd_cuda<float, at::Half>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_bwd_cuda<at::Half, at::Half>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_bwd_cuda<at::BFloat16, at::Half>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_bwd_cuda<float, at::BFloat16>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_bwd_cuda<at::Half, at::BFloat16>(ConvParamsBwd &params, cudaStream_t stream);
template void causal_conv1d_bwd_cuda<at::BFloat16, at::BFloat16>(ConvParamsBwd &params, cudaStream_t stream);

