#pragma once
#include "kernel_traits.cuh"
#include "utils.cuh"
#include "config.h"
#include "softmax.cuh"

namespace fa2 {
using namespace cute;

template <typename Config>
__global__ void flash_attn_fwd_kernel(FlashAttnConfig config) {
    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kHeadDim = Config::kHeadDim;

    using GmemCopyQKV = typename Config::GmemCopyQKV;
    using GmemCopyO = typename Config::GmemCopyO;
    using SmemCopyAtom = typename Config::SmemCopyAtom;
    using SmemCopyAtomO = typename Config::SmemCopyAtomO;
    using SmemCopyAtomT = typename Config::SmemCopyAtomT;
    using MMA = typename Config::MMA;
    using SmemLayoutQ = typename Config::SmemLayoutQ;
    using SmemLayoutKV = typename Config::SmemLayoutKV;
    using SmemLayoutVt = typename Config::SmemLayoutVt;
    using SmemLayoutVtNoSwizzle = typename Config::SmemLayoutVtNoSwizzle;
    using SmemLayoutO = typename Config::SmemLayoutO;

    // q chunk
    int bix = blockIdx.x;
    // batch × head
    int biy = blockIdx.y;
    int batch_idx = biy / config.num_heads;
    int head_idx = biy % config.num_heads;
    int tid = threadIdx.x;

    extern __shared__ __align__(16) half_t smem[];

    Tensor mQ = make_tensor(
        make_gmem_ptr(reinterpret_cast<const half_t*>(config.q_ptr) + batch_idx * config.q_batch_stride + head_idx * config.q_head_stride),
        make_shape(config.seq_len_q, config.head_dim),
        make_stride(config.q_seq_stride, _1{})
    );

    Tensor mK = make_tensor(
        make_gmem_ptr(reinterpret_cast<const half_t*>(config.k_ptr) + batch_idx * config.k_batch_stride + head_idx * config.k_head_stride),
        make_shape(config.seq_len_kv, config.head_dim),
        make_stride(config.k_seq_stride, _1{})
    );

    Tensor mV = make_tensor(
        make_gmem_ptr(reinterpret_cast<const half_t*>(config.v_ptr) + batch_idx * config.v_batch_stride + head_idx * config.v_head_stride),
        make_shape(config.seq_len_kv, config.head_dim),
        make_stride(config.v_seq_stride, _1{})
    );

    // [kTileM, kHeadDim]
    Tensor gQ = local_tile(mQ, make_shape(Int<kTileM>{}, Int<kHeadDim>{}), make_coord(bix, 0));
    // [kTileN, kHeadDim, num_n_tiles]
    Tensor gK = local_tile(mK, make_shape(Int<kTileN>{}, Int<kHeadDim>{}), make_coord(_, 0));
    // [kTileN, kHeadDim, num_n_tiles]
    Tensor gV = local_tile(mV, make_shape(Int<kTileN>{}, Int<kHeadDim>{}), make_coord(_, 0));

    // [kTileM, kHeadDim]
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<half_t*>(smem)), SmemLayoutQ{});
    // [kTileN, kHeadDim]
    Tensor sK = make_tensor(make_smem_ptr(sQ.data() + size(sQ)), SmemLayoutKV{});
    // [kTileN, kHeadDim]
    Tensor sV = make_tensor(make_smem_ptr(sK.data() + size(sK)), SmemLayoutKV{});
    // [kHeadDim, kTileN]
    Tensor sVt = make_tensor(make_smem_ptr(sV.data()), SmemLayoutVt{});
    // [kHeadDim, kTileN]
    Tensor sVtNoSwizzle = make_tensor(make_smem_ptr(sV.data()), SmemLayoutVtNoSwizzle{});
    
    MMA tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    // [MMA, MMA_M, MMA_H]
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    // [MMA, MMA_N, MMA_H]
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor acc_s = partition_fragment_C(tiled_mma, make_shape(Int<kTileM>{}, Int<kTileN>{}));
    // [MMA, MMA_N, MMA_H]
    Tensor tOrV = thr_mma.partition_fragment_B(sVtNoSwizzle);
    clear(acc_s);

    GmemCopyQKV tiled_copy_qkv;
    auto thr_copy_qkv = tiled_copy_qkv.get_thread_slice(tid);
    // [CPY, CPY_M, CPY_H]
    Tensor tQgQ = thr_copy_qkv.partition_S(gQ);
    // [CPY, CPY_M, CPY_H]
    Tensor tQsQ = thr_copy_qkv.partition_D(sQ);
    // [CPY, CPY_N, CPY_H, num_k_tiles]
    Tensor tKgK = thr_copy_qkv.partition_S(gK);
    // [CPY, CPY_N, CPY_H]
    Tensor tKsK = thr_copy_qkv.partition_D(sK);
    // [CPY, CPY_N, CPY_H, num_k_tiles]
    Tensor tVgV = thr_copy_qkv.partition_S(gV);
    // [CPY, CPY_N, CPY_H]
    Tensor tVsV = thr_copy_qkv.partition_D(sV);

    auto smem_copy_q = make_tiled_copy_A(
        SmemCopyAtom{},
        tiled_mma
    );
    auto smem_copy_k = make_tiled_copy_B(
        SmemCopyAtom{},
        tiled_mma
    );
    auto smem_copy_v = make_tiled_copy_B(
        SmemCopyAtomT{},
        tiled_mma
    );
    auto smem_thr_copy_q = smem_copy_q.get_thread_slice(tid);
    auto smem_thr_copy_k = smem_copy_k.get_thread_slice(tid);
    auto smem_thr_copy_v = smem_copy_v.get_thread_slice(tid);


    // [CPY, CPY_M, CPY_H]
    Tensor tSsQ = smem_thr_copy_q.partition_S(sQ);
    // [CPY, CPY_N, CPY_H]
    Tensor tSsK = smem_thr_copy_k.partition_S(sK);
    // [CPY, CPY_N, CPY_H]
    Tensor tOsVt = smem_thr_copy_v.partition_S(sVt);
    Tensor acc_o = partition_fragment_C(
      tiled_mma, make_shape(Int<kTileM>{}, Int<kHeadDim>{}));
    clear(acc_o);

    // Single K and V buffers: overlap V copy with QK, then next K with PV.
    copy(tiled_copy_qkv, tQgQ, tQsQ);
    copy(tiled_copy_qkv, tKgK(_, _, _, _0{}), tKsK);
    cp_async_fence();

    Softmax<2 * size<1>(acc_o)> softmax;

    int n_block_n = ceil_div(config.seq_len_kv, kTileN);
    for (int n = 0; n < n_block_n; ++n) {
        clear(acc_s);
        cp_async_wait<0>();
        __syncthreads();
        // 发射 V tile copy
        copy(tiled_copy_qkv, tVgV(_, _, _, n), tVsV);
        cp_async_fence();

        fa2::gemm(
            acc_s, 
            tSrQ, 
            tSrK, 
            tSsQ, 
            tSsK, 
            tiled_mma, 
            smem_copy_q,
            smem_copy_k,
            smem_thr_copy_q,
            smem_thr_copy_k
        );

        // 等待 V 结束
        cp_async_wait<0>();
        __syncthreads();

        // 预取 K tile
        if (n < n_block_n - 1) {
            copy(tiled_copy_qkv, tKgK(_, _, _, n + 1), tKsK);
            cp_async_fence();
        }

        if (n == 0) {
            softmax.template softmax_rescale_o<true>(
                acc_s, acc_o, config.scale_softmax_log2
            );
        } else {
            softmax.template softmax_rescale_o<false>(
                acc_s, acc_o, config.scale_softmax_log2
            );
        }

        Tensor acc_s_fp16 = convert_type<half_t>(acc_s);
        Tensor tOrP = make_tensor(acc_s_fp16.data(), convert_c_frag_to_a_frag(acc_s_fp16.layout()));

        gemm_rs(acc_o, tOrP, tOrV, tOsVt, tiled_mma, smem_copy_v, smem_thr_copy_v);
    }

    softmax.normalize_softmax(acc_o);

    Tensor o_fp16 = convert_type<half_t>(acc_o);
    Tensor sO = make_tensor(sQ.data(), SmemLayoutO{});
    auto smem_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_copy_O.get_thread_slice(tid);
    auto trO = smem_thr_copy_O.retile_S(o_fp16);
    auto tsO = smem_thr_copy_O.partition_D(sO);
    copy(smem_copy_O, trO, tsO);

    GmemCopyO gmem_copy_O;
    Tensor mO = make_tensor(reinterpret_cast<half_t*>(config.o_ptr) + batch_idx * config.o_batch_stride + head_idx * config.o_head_stride,
                            make_shape(config.seq_len_q, config.head_dim),
                            make_stride(config.o_seq_stride, _1{}));

    Tensor gO = local_tile(mO, make_shape(Int<kTileM>{}, Int<kHeadDim>{}), make_coord(bix, 0));
    auto gmem_thr_copy_O = gmem_copy_O.get_thread_slice(tid);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    Tensor tOrO = make_fragment_like(tOgO);
    __syncthreads();
    copy(gmem_copy_O, tOsO, tOrO);
    copy(gmem_copy_O, tOrO, tOgO);
}
}
