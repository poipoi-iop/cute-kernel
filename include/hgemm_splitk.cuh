#pragma once

// Split-K FP16 GEMM in the style of include/gemm.cuh, following docs/splitk.md.
//
// Contract: C[M,N] = A[M,K] * transpose(B[N,K]), all row-major FP16.
//
// The three differences from gemm.cuh, and nothing else:
//   1. grid has a third dimension: blockIdx.z indexes a K segment
//   2. the K loop runs over [k_tile_begin, k_tile_end) instead of [0, k_tiles)
//   3. the epilogue writes FP32 partials to a workspace, and a second kernel
//      reduces them into C
//
// Like gemm.cuh, this assumes whole tiles: M and N must be multiples of the
// CTA tile and K a multiple of BK, matching the 128-bit cp.async requirement.
// can_launch() below reports that, and the caller falls back otherwise.
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cstddef>

namespace hgemm_splitk {

using namespace cute;

// ---------------------------------------------------------------------------
// Tile shapes and the CuTe types of one tile. Same shape of Config as in
// gemm.cuh: the kernel below only ever asks Config for a type or a constant.
// ---------------------------------------------------------------------------
using T = half_t;

using BM_t = Int<64>;  // CTA tile rows
using BN_t = Int<64>;  // CTA tile cols
using BK_t = Int<32>;  // K width of one pipeline stage
constexpr int kStages = 3;
constexpr int kThreads = 128;

using MmaOp = SM80_16x8x16_F32F16F16F32_TN;
using TiledMma = decltype(make_tiled_mma(
    MmaOp{},
    make_layout(make_shape(_2{}, _2{}, _1{})),   // 4 warps: 2 in M, 2 in N
    make_tile(_32{}, _32{}, _16{})));            // 32x32x16 per warp

// Global -> shared, 16 bytes (8 halves) per thread.
using G2SCopy = decltype(make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, T>{},
    make_layout(make_shape(_32{}, _4{}), make_stride(_4{}, _1{})),
    make_layout(make_shape(_1{}, _8{}))));

// Shared: 8x32 atom with a swizzle, tiled over (BM|BN, BK, stages).
using SmemAtom = decltype(composition(
    Swizzle<3, 3, 3>{},
    make_layout(make_shape(_8{}, BK_t{}), make_stride(BK_t{}, _1{}))));
using SmemLayoutA = decltype(tile_to_shape(
    SmemAtom{}, make_shape(BM_t{}, BK_t{}, Int<kStages>{})));
using SmemLayoutB = decltype(tile_to_shape(
    SmemAtom{}, make_shape(BN_t{}, BK_t{}, Int<kStages>{})));

// Shared -> registers via ldmatrix.
using S2RCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, T>;
using S2RCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, T>;

constexpr int kSmemBytes = (cosize(SmemLayoutA{}) + cosize(SmemLayoutB{})) * sizeof(T);

// ---------------------------------------------------------------------------
// Main kernel. Only the K interval and the epilogue differ from gemm.cuh.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(kThreads) void hgemm_splitk_kernel(
    T const* __restrict__ A,       // [M, K] row-major
    T const* __restrict__ B,       // [N, K] row-major, i.e. B^T
    float* __restrict__ ws,        // [splits, M, N] FP32
    T* __restrict__ C,             // [M, N] only written when splits == 1
    int M, int N, int K, int splits) {
    constexpr int BK = BK_t::value;
    constexpr int BM = BM_t::value;
    constexpr int BN = BN_t::value;

    // This CTA's K interval, in units of BK. Contiguous and disjoint; the
    // trailing splits can be empty when K does not divide evenly.
    const int k_tiles = K / BK;
    const int per_split = (k_tiles + splits - 1) / splits;
    const int k_begin = blockIdx.z * per_split;
    const int k_count = min(per_split, k_tiles - k_begin);

    // Same convention as gemm.cuh: blockIdx.x walks N, blockIdx.y walks M.
    const int ix = blockIdx.x, iy = blockIdx.y;

    // gA: (BM, BK, k_tiles), gB: (BN, BK, k_tiles). The k-tile mode is what the
    // pipeline steps through; k_begin is picked per iteration.
    Tensor mA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(M, K), make_stride(K, _1{})));
    Tensor mB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(N, K), make_stride(K, _1{})));
    Tensor gA = local_tile(mA, make_shape(BM_t{}, BK_t{}), make_coord(iy, _));
    Tensor gB = local_tile(mB, make_shape(BN_t{}, BK_t{}), make_coord(ix, _));

    extern __shared__ __align__(16) char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);
    Tensor sA = make_tensor(make_smem_ptr(smem), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem + cosize(SmemLayoutA{})), SmemLayoutB{});

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    auto thr_g2s_a = G2SCopy{}.get_slice(threadIdx.x);
    auto thr_g2s_b = G2SCopy{}.get_slice(threadIdx.x);
    Tensor tAgA = thr_g2s_a.partition_S(gA);   // (CPY, CPY_M, CPY_K, k_tiles)
    Tensor tAsA = thr_g2s_a.partition_D(sA);   // (CPY, CPY_M, CPY_K, stages)
    Tensor tBgB = thr_g2s_b.partition_S(gB);
    Tensor tBsB = thr_g2s_b.partition_D(sB);

    auto s2r_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto thr_s2r_a = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tCsA = thr_s2r_a.partition_S(sA);
    // Fragment layouts come off the logical (M,K) gmem tile, exactly as in
    // gemm.cuh. Deriving them from the swizzled smem tensor would describe the
    // bank-conflict-free address pattern instead of the MMA's register layout.
    Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    Tensor rA = thr_s2r_a.retile_D(tCrA);

    auto s2r_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto thr_s2r_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tCsB = thr_s2r_b.partition_S(sB);
    Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    Tensor rB = thr_s2r_b.retile_D(tCrB);

    Tensor accum = partition_fragment_C(tiled_mma, make_shape(BM_t{}, BN_t{}));
    clear(accum);

    // An empty K interval contributes zeros, not a skipped store: every split
    // owns its own FP32 slab and every element of it must be written.
    if (k_count <= 0) {
        float* slab = ws + (std::size_t)blockIdx.z * M * N;
        Tensor gWS = local_tile(
            make_tensor(make_gmem_ptr(slab), make_layout(make_shape(M, N), make_stride(N, _1{}))),
            make_shape(BM_t{}, BN_t{}), make_coord(iy, ix));
        Tensor tCgWS = thr_mma.partition_C(gWS);
#pragma unroll
        for (int i = 0; i < size(accum); ++i) tCgWS(i) = accum(i);
        return;
    }

    // Prologue: kStages-1 tiles in flight. The group count matters as much as
    // the data: the consumer below waits a fixed kStages-2 groups per iteration,
    // so a short K interval has to commit the same number of groups, using empty
    // ones for the tiles that do not exist.
    int smem_wr = 0;
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < k_count) {
            copy(G2SCopy{}, tAgA(_, _, _, k_begin + s), tAsA(_, _, _, smem_wr));
            copy(G2SCopy{}, tBgB(_, _, _, k_begin + s), tBsB(_, _, _, smem_wr));
            smem_wr = (smem_wr + 1) % kStages;
        }
        cp_async_fence();
    }

    // Steady state.
    int smem_rd = 0;
    for (int k = 0; k < k_count; ++k) {
        cp_async_wait<kStages - 2>();
        __syncthreads();

        // rA/rB are the ldmatrix-shaped views of the MMA fragments: the copies
        // write through them, gemm() reads the same registers as tCrA/tCrB.
        copy(s2r_copy_a, tCsA(_, _, _, smem_rd), rA);
        copy(s2r_copy_b, tCsB(_, _, _, smem_rd), rB);

        if (k + kStages - 1 < k_count) {
            copy(G2SCopy{}, tAgA(_, _, _, k_begin + k + kStages - 1), tAsA(_, _, _, smem_wr));
            copy(G2SCopy{}, tBgB(_, _, _, k_begin + k + kStages - 1), tBsB(_, _, _, smem_wr));
            smem_wr = (smem_wr + 1) % kStages;
        }
        cp_async_fence();

        gemm(tiled_mma, accum, tCrA, tCrB, accum);
        smem_rd = (smem_rd + 1) % kStages;
    }
    cp_async_wait<0>();
    __syncthreads();

    // Epilogue: the accumulator is partitioned against the output tile, so the
    // same partition that addresses C also addresses this split's slab.
    if (splits == 1) {
        Tensor gC = local_tile(
            make_tensor(make_gmem_ptr(C), make_layout(make_shape(M, N), make_stride(N, _1{}))),
            make_shape(BM_t{}, BN_t{}), make_coord(iy, ix));
        Tensor tCgC = thr_mma.partition_C(gC);
#pragma unroll
        for (int i = 0; i < size(accum); ++i) tCgC(i) = static_cast<T>(accum(i));
    } else {
        Tensor gWS = local_tile(
            make_tensor(make_gmem_ptr(ws + (std::size_t)blockIdx.z * M * N),
                make_layout(make_shape(M, N), make_stride(N, _1{}))),
            make_shape(BM_t{}, BN_t{}), make_coord(iy, ix));
        Tensor tCgWS = thr_mma.partition_C(gWS);
#pragma unroll
        for (int i = 0; i < size(accum); ++i) tCgWS(i) = accum(i);
    }
}

// ---------------------------------------------------------------------------
// Reduce: one thread owns 4 consecutive outputs, so each pass over the
// workspace is a 16-byte load and the store is two half2.
// ---------------------------------------------------------------------------
__global__ void hgemm_splitk_reduce_kernel(
    float const* __restrict__ ws, T* __restrict__ C, int splits, int MN) {
    const int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base >= MN) return;

    float4 sum = *reinterpret_cast<float4 const*>(&ws[base]);
    for (int z = 1; z < splits; ++z) {
        const float4 part = *reinterpret_cast<float4 const*>(&ws[(std::size_t)z * MN + base]);
        sum.x += part.x; sum.y += part.y; sum.z += part.z; sum.w += part.w;
    }
    *reinterpret_cast<half2*>(&C[base]) = __float22half2_rn(make_float2(sum.x, sum.y));
    *reinterpret_cast<half2*>(&C[base + 2]) = __float22half2_rn(make_float2(sum.z, sum.w));
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

// Whole tiles only, as in gemm.cuh.
inline bool can_launch(int M, int N, int K) {
    return M > 0 && N > 0 && K > 0 && M % BM_t::value == 0 && N % BN_t::value == 0 &&
        K % BK_t::value == 0;
}

// Workspace is [splits, M, N] FP32. Splits of 1 write C directly and need none.
inline std::size_t hgemm_splitk_workspace_bytes(int M, int N, int splits) {
    return (std::size_t)4 * splits * M * N;
}

inline cudaError_t hgemm_splitk_launch(int M, int N, int K, T const* A, T const* B, T* C,
    float* ws, int splits = 1, cudaStream_t stream = nullptr) {
    if (!can_launch(M, N, K)) return cudaErrorNotSupported;
    if (splits < 1 || splits > 65535) return cudaErrorInvalidValue;
    if (!A || !B || !C) return cudaErrorInvalidValue;
    if (splits > 1 && !ws) return cudaErrorInvalidValue;

    static bool configured = false;
    if (!configured) {
        auto status = cudaFuncSetAttribute(
            hgemm_splitk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
        if (status != cudaSuccess) return status;
        configured = true;
    }
    // blockIdx.x walks N and blockIdx.y walks M, matching the kernel.
    dim3 grid(N / BN_t::value, M / BM_t::value, splits);
    hgemm_splitk_kernel<<<grid, kThreads, kSmemBytes, stream>>>(A, B, ws, C, M, N, K, splits);
    auto status = cudaGetLastError();
    if (status != cudaSuccess || splits == 1) return status;

    const int MN = M * N;
    hgemm_splitk_reduce_kernel<<<(MN / 4 + 255) / 256, 256, 0, stream>>>(ws, C, splits, MN);
    return cudaGetLastError();
}

}  // namespace hgemm_splitk
