#pragma once

// New implementation derived from include/gemm.cuh's tiled CuTe design.
// Contract: C[M,N] = A[M,K] * transpose(B[N,K]), all row-major FP16.
// SM80+, FP32 accumulation; no allocation or synchronization in launch().
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cute_kernel {

// 一个 CTA 使用 4 个 warp，按 2×2 的 warp 网格覆盖 M/N。
// MMA 指令的输入为 FP16，accumulator 为 FP32；TileM/TileN 必须能被
// warp 网格的基本 tile 整除。K 固定为 32，即每轮两组 k=16 的 MMA。
template <int TileM = 64, int TileN = 64, int Stages = 3>
struct SplitKConfig {
    static_assert(TileM % 32 == 0 && TileN % 32 == 0);
    static_assert(Stages >= 2 && Stages <= 4);
    using T = cute::half_t;
    static constexpr int kTileM = TileM, kTileN = TileN, kTileK = 32;
    static constexpr int kStages = Stages, kThreads = 128;
    using TiledMMA = decltype(cute::make_tiled_mma(
            cute::SM80_16x8x16_F32F16F16F32_TN{},
            cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>{},
            cute::Tile<cute::_32, cute::_32, cute::_16>{}));
    // swizzle 改变 shared-memory bank 映射，而不改变逻辑 (row,k,stage)
    // 坐标。低 3 位保持连续，因此一个 8-half 向量仍可用 16-byte cp.async。
    using SmemAtom = decltype(cute::composition(
            cute::Swizzle<2, 3, 3>{},
            cute::Layout<cute::Shape<cute::_8, cute::_32>,
            cute::Stride<cute::_32, cute::_1>>{}));
    using SmemA = decltype(cute::tile_to_shape(
            SmemAtom{}, cute::Shape<cute::Int<TileM>, cute::_32, cute::Int<Stages>>{}));
    using SmemB = decltype(cute::tile_to_shape(
            SmemAtom{}, cute::Shape<cute::Int<TileN>, cute::_32, cute::Int<Stages>>{}));
    static constexpr int kSharedBytes =
        (cute::cosize(SmemA{}) + cute::cosize(SmemB{})) * sizeof(T);
};

namespace splitk_detail {

inline int ceil_div(int x, int y) { return x / y + (x % y != 0); }

// A complete 16-byte vector is copied asynchronously. Invalid vectors use
// cp.async's zero-fill operand; an unaligned K uses the scalar fallback below.
__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(address), "l"(src), "r"(valid ? 16 : 0) : "memory");
#endif
}

// 将一个 K tile 写入指定 ring-buffer stage，所有元素都必须被覆盖。
// 完整向量使用 cp.async，超出矩阵或当前 split 的向量写零；fallback 对
// 每个元素做 predicate。这样短 K、尾部和流水线 drain 不会读取遗留数据。
template <class Config, bool Vectorized, class TensorA, class TensorB>
__device__ __forceinline__ void load_tile(
    const typename Config::T* a, const typename Config::T* b,
    TensorA& sa, TensorB& sb, int m, int n, int k, int row_a, int row_b,
    int ktile, int end_tile, int stage) {
    using T = typename Config::T;
    constexpr int BK = Config::kTileK;
    if constexpr (Vectorized) {
#pragma unroll
        for (int v = threadIdx.x; v < Config::kTileM * BK / 8; v += Config::kThreads) {
            int r = v / (BK / 8), c = (v % (BK / 8)) * 8;
            std::int64_t gk = std::int64_t(ktile) * BK + c;
            bool valid = ktile < end_tile && row_a + r < m && gk < k;
            const T* source = valid ? a + std::size_t(row_a + r) * k + gk : a;
            copy16(&sa(r, c, stage), source, valid);
        }
#pragma unroll
        for (int v = threadIdx.x; v < Config::kTileN * BK / 8; v += Config::kThreads) {
            int r = v / (BK / 8), c = (v % (BK / 8)) * 8;
            std::int64_t gk = std::int64_t(ktile) * BK + c;
            bool valid = ktile < end_tile && row_b + r < n && gk < k;
            const T* source = valid ? b + std::size_t(row_b + r) * k + gk : b;
            copy16(&sb(r, c, stage), source, valid);
        }
    } else {
        // General strides and K tails: each shared-memory element is initialized.
#pragma unroll
        for (int i = threadIdx.x; i < Config::kTileM * BK; i += Config::kThreads) {
            int r = i / BK, c = i % BK;
            std::int64_t gk = std::int64_t(ktile) * BK + c;
            bool valid = ktile < end_tile && row_a + r < m && gk < k;
            sa(r, c, stage) = valid ? a[std::size_t(row_a + r) * k + gk] : T(0);
        }
#pragma unroll
        for (int i = threadIdx.x; i < Config::kTileN * BK; i += Config::kThreads) {
            int r = i / BK, c = i % BK;
            std::int64_t gk = std::int64_t(ktile) * BK + c;
            bool valid = ktile < end_tile && row_b + r < n && gk < k;
            sb(r, c, stage) = valid ? b[std::size_t(row_b + r) * k + gk] : T(0);
        }
    }
    cute::cp_async_fence();
}

}  // namespace splitk_detail

// Parallel Split-K: each CTA owns an independent K interval, writes FP32
// partials, and a second kernel reduces them in a fixed order. S=1 writes C
// directly. No atomics, half partials, workspace memset, or inter-CTA spinlocks.
template <class Config = SplitKConfig<>, bool Vectorized = true, bool Split = true>
__global__ __launch_bounds__(Config::kThreads)
void hgemm_splitk(cute::half_t* __restrict__ c,
    const cute::half_t* __restrict__ a,
    const cute::half_t* __restrict__ b,
    float* __restrict__ partials, int m, int n, int k, int splits) {
    using namespace cute;
    constexpr int BM = Config::kTileM, BN = Config::kTileN, BK = Config::kTileK;
    constexpr int S = Config::kStages;
    // 将 M/N tile 网格展平到 grid.x，grid.z 专门表示 K 分区。
    // begin/end 是 tile 索引，不是元素索引；整数比例切分允许不均匀分区，
    // 相邻分区的边界相接，完整覆盖 [0,ceil(K/BK)) 且没有重叠。
    int tiles_n = n / BN + (n % BN != 0);
    int row_a = (blockIdx.x / tiles_n) * BM;
    int row_b = (blockIdx.x % tiles_n) * BN;
    int total_tiles = k / BK + (k % BK != 0);
    int begin = int(std::int64_t(total_tiles) * blockIdx.z / splits);
    int end = int(std::int64_t(total_tiles) * (blockIdx.z + 1) / splits);

    // A/B 分别在动态 shared memory 中持有 S 个 stage；cosize 考虑了
    // CuTe layout 的实际寻址范围。16-byte 对齐满足异步拷贝要求。
    extern __shared__ __align__(16) unsigned char storage[];
    auto* raw = reinterpret_cast<half_t*>(storage);
    auto sa = make_tensor(make_smem_ptr(raw), typename Config::SmemA{});
    auto sb = make_tensor(make_smem_ptr(raw + cosize(typename Config::SmemA{})),
        typename Config::SmemB{});
    // 每个线程只持有自身的 operand/accumulator fragment。
    // make_tiled_copy_A/B + retile_D 连接 ldmatrix 的拷贝布局与 MMA 布局，
    // 不能假设连续线程各自拥有连续的输出元素。
    typename Config::TiledMMA mma;
    auto thread_mma = mma.get_slice(threadIdx.x);
    auto ra = thread_mma.partition_fragment_A(sa(_, _, 0));
    auto rb = thread_mma.partition_fragment_B(sb(_, _, 0));
    auto rc = partition_fragment_C(mma, Shape<Int<BM>, Int<BN>>{});
    clear(rc);
    auto copy_a = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, mma);
    auto copy_b = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, half_t>{}, mma);
    auto ta = copy_a.get_slice(threadIdx.x);
    auto tb = copy_b.get_slice(threadIdx.x);
    auto from_a = ta.partition_S(sa);
    auto from_b = tb.partition_S(sb);
    auto to_a = ta.retile_D(ra);
    auto to_b = tb.retile_D(rb);

    // Keep exactly S-1 committed groups ahead. Even the drain commits zero-filled
    // groups, so wait_group<S-2> always waits for the stage about to be consumed.
#pragma unroll
    for (int s = 0; s < S - 1; ++s)
        splitk_detail::load_tile<Config, Vectorized>(
            a, b, sa, sb, m, n, k, row_a, row_b, begin + s, end, s);

    for (int tile = begin; tile < end; ++tile) {
        int read_stage = (tile - begin) % S;
        cp_async_wait<S - 2>();
        __syncthreads();
        // The spare stage overlaps global-memory traffic with the current MMA.
        int write_stage = (tile - begin + S - 1) % S;
        splitk_detail::load_tile<Config, Vectorized>(
            a, b, sa, sb, m, n, k, row_a, row_b, tile + S - 1, end, write_stage);
        copy(copy_a, from_a(_, _, _, read_stage), to_a);
        copy(copy_b, from_b(_, _, _, read_stage), to_b);
#pragma unroll
        for (int kk = 0; kk < size<2>(ra); ++kk)
            gemm(mma, rc, ra(_, _, kk), rb(_, _, kk), rc);
        __syncthreads();  // all readers finish before a stage can be reused
    }
    cp_async_wait<0>();
    __syncthreads();

    // 对 identity tensor 做与 accumulator 一样的 partition，得到每个
    // rc(i) 对应的真实 (M,N) 坐标，再逐元素做尾部 predicate。
    // Split-K 路径先存 FP32，避免每个分区各自舍入为 FP16 后再相加。
    auto coords = thread_mma.partition_C(make_identity_tensor(Shape<Int<BM>, Int<BN>>{}));
#pragma unroll
    for (int i = 0; i < size(rc); ++i) {
        int r = row_a + get<0>(coords(i)), col = row_b + get<1>(coords(i));
        if (r < m && col < n) {
            std::size_t offset = std::size_t(r) * n + col;
            if constexpr (Split)
                partials[std::size_t(blockIdx.z) * m * n + offset] = rc(i);
            else
                c[offset] = half_t(rc(i));
        }
    }
}

namespace splitk_detail {
// workspace 为 [splits,M*N]；每个输出由一个线程按固定 split 顺序求和。
// 相邻线程访问相邻输出，因此每轮 split 的读取都是合并访问。
static __global__ void reduce(cute::half_t* out, const float* partials,
    std::size_t count, int splits) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        i < count; i += std::size_t(blockDim.x) * gridDim.x) {
        float sum = 0;
        for (int s = 0; s < splits; ++s) sum += partials[std::size_t(s) * count + i];
        out[i] = cute::half_t(sum);
    }
}
}  // namespace splitk_detail

// 可复用的 host launch 参数。workspace 不在 plan 内分配或持有，避免
// 把分配、初始化或 device 查询纳入 kernel 时间及 CUDA Graph capture。
struct HgemmSplitKPlan {
    int m = 0, n = 0, k = 0, splits = 1;
    std::size_t workspace_bytes = 0;
};

// sm_count is explicit: device queries belong in setup, outside CUDA graphs.
// requested_splits=0 uses a conservative occupancy heuristic; tune externally.
inline cudaError_t make_hgemm_splitk_plan(HgemmSplitKPlan* plan, int m, int n,
    int k, int requested_splits = 0,
    int sm_count = 80) {
    if (!plan || m < 0 || n < 0 || k < 0 || requested_splits < 0 || sm_count <= 0)
        return cudaErrorInvalidValue;
    int tiles_k = splitk_detail::ceil_div(k, 32);
    std::size_t blocks = std::size_t(splitk_detail::ceil_div(m, 64)) *
        splitk_detail::ceil_div(n, 64);
    if (blocks > 2147483647ULL) return cudaErrorInvalidConfiguration;
    int splits = requested_splits;
    if (splits == 0) {
        // 目标约为每个 SM 两个输出 CTA，每个 split 至少保留约 8 个 K tile。
        // 这是并行度启发式；大 M/N 通常选择 splits=1，跳过额外归约。
        std::size_t target = std::size_t(sm_count) * 2;
        int needed = int((target + std::max<std::size_t>(1, blocks) - 1) /
            std::max<std::size_t>(1, blocks));
        splits = std::max(1, std::min({needed, 32, tiles_k / 8}));
    }
    if (splits > 65535 || (k > 0 && splits > tiles_k)) return cudaErrorInvalidValue;
    if (!k || !m || !n) splits = 1;
    std::size_t count = std::size_t(m) * n;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float) / splits)
        return cudaErrorInvalidValue;
    *plan = {m, n, k, splits, splits > 1 ? count * splits * sizeof(float) : 0};
    return cudaSuccess;
}

inline cudaError_t launch_hgemm_splitk(
    const HgemmSplitKPlan& plan, cute::half_t* c, const cute::half_t* a,
    const cute::half_t* b, void* workspace = nullptr, std::size_t workspace_bytes = 0,
    cudaStream_t stream = nullptr) {
    // 公开 plan 可被调用方修改，因此 launch 必须重新检查维度、split 数
    // 和 workspace 大小。这里只检查可在 host 判断的契约，不查询设备。
    HgemmSplitKPlan checked;
    auto status = make_hgemm_splitk_plan(&checked, plan.m, plan.n, plan.k, plan.splits);
    if (status != cudaSuccess || plan.splits != checked.splits ||
        plan.workspace_bytes != checked.workspace_bytes) return cudaErrorInvalidValue;
    if (!plan.m || !plan.n) return cudaSuccess;
    if (!c || reinterpret_cast<std::uintptr_t>(c) % alignof(cute::half_t))
        return cudaErrorInvalidValue;
    if (!plan.k) return cudaMemsetAsync(c, 0, std::size_t(plan.m) * plan.n * sizeof(*c), stream);
    if (!a || !b || reinterpret_cast<std::uintptr_t>(a) % alignof(cute::half_t) ||
        reinterpret_cast<std::uintptr_t>(b) % alignof(cute::half_t) ||
        (checked.workspace_bytes && (!workspace || workspace_bytes < checked.workspace_bytes ||
                reinterpret_cast<std::uintptr_t>(workspace) % alignof(float))))
        return cudaErrorInvalidValue;
    using Config = SplitKConfig<>;
    unsigned blocks = unsigned(std::size_t(splitk_detail::ceil_div(plan.m, 64)) *
        splitk_detail::ceil_div(plan.n, 64));
    dim3 grid(blocks, 1, plan.splits);
    // 每条向量读 8 个 half；行 stride 与两个首地址均对齐才使用 cp.async。
    // 输出无需 16-byte 对齐，因为 epilogue 使用按元素 predicate 的 store。
    bool vectorized = plan.k % 8 == 0 &&
        (reinterpret_cast<std::uintptr_t>(a) | reinterpret_cast<std::uintptr_t>(b)) % 16 == 0;
#define CK_SPLITK_LAUNCH(V, S) \
    hgemm_splitk<Config, V, S><<<grid, Config::kThreads, Config::kSharedBytes, stream>>>( \
        c, a, b, static_cast<float*>(workspace), plan.m, plan.n, plan.k, plan.splits)
    if (plan.splits == 1) {
        if (vectorized) { CK_SPLITK_LAUNCH(true, false); }
        else { CK_SPLITK_LAUNCH(false, false); }
    } else {
        if (vectorized) { CK_SPLITK_LAUNCH(true, true); }
        else { CK_SPLITK_LAUNCH(false, true); }
    }
#undef CK_SPLITK_LAUNCH
    status = cudaGetLastError();
    if (status != cudaSuccess || plan.splits == 1) return status;
    std::size_t count = std::size_t(plan.m) * plan.n;
    unsigned reduce_blocks = unsigned(std::min<std::size_t>((count + 255) / 256, 65535));
    splitk_detail::reduce<<<reduce_blocks, 256, 0, stream>>>(
        c, static_cast<const float*>(workspace), count, plan.splits);
    return cudaGetLastError();
}

}  // namespace cute_kernel
