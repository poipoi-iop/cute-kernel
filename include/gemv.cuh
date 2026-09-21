#pragma once

// CuTe GEMV: y[M] = A[M,K] x[K]，A 为连续行主序。
//
// 数据流：CuTe global tensor -> TiledCopy 的线程分片 -> register tensor
//         -> FP32 FMA -> warp subgroup 归约 -> y 或 FP32 Split-K workspace。
// 输入/输出支持 __half 和 float；累加与跨分区归约始终使用 FP32。
// CuTe 负责布局和拷贝，行内归约使用 CUDA shuffle，不把 GEMV 扩展成稠密 GEMM。
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cute_kernel {
namespace gemv_detail {

// 将存储类型与累加类型分离。输出只在最后一次 store 前做 round-to-nearest。
template <class T>
__device__ __forceinline__ float as_float(T x) { return float(x); }
template <>
__device__ __forceinline__ float as_float(__half x) { return __half2float(x); }
template <class T>
__device__ __forceinline__ T from_float(float x) { return T(x); }
template <>
__device__ __forceinline__ __half from_float(float x) { return __float2half_rn(x); }

// 一个 CTA 覆盖 Rows 行；每行分配 Group 个相邻 lane。
// ThreadLayout 的 stride=(Group,1)，因此 tid=row*Group+lane。
// ValueLayout 给每个线程分配同一行上连续的 V 个元素，整个 copy tile
// 的 shape 为 (Rows, Group*V)。FP16 的 V=8，FP32 的 V=4。
// 向量分支明确使用 128-bit Copy_Atom；fallback 的 V=1，避免未对齐访问。
template <class T, int Group, int Warps, bool Vectorized>
struct CopyConfig {
    static constexpr int kThreads = Warps * 32;
    static constexpr int kRows = kThreads / Group;
    static constexpr int kValues = Vectorized ? 16 / sizeof(T) : 1;
    using ThreadLayout = cute::Layout<
        cute::Shape<cute::Int<kRows>, cute::Int<Group>>,
        cute::Stride<cute::Int<Group>, cute::_1>>;
    using ValueLayout = cute::Layout<cute::Shape<cute::_1, cute::Int<kValues>>>;
    using CopyWord = std::conditional_t<Vectorized, cute::uint128_t, T>;
    using CopyAtom = cute::Copy_Atom<cute::UniversalCopy<CopyWord>, T>;
    using TiledCopy = decltype(cute::make_tiled_copy(
            CopyAtom{}, ThreadLayout{}, ValueLayout{}));
    using Tile = cute::Shape<cute::Int<kRows>, cute::Int<Group * kValues>>;
};

// fragment 中元素已经由 CuTe 放入寄存器。FP16 先转换成 float，再做 FMA，
// 避免 half 乘加累积误差。四条独立依赖链在最后合并，提供指令级并行。
template <class T, int Values, class FragmentA, class FragmentX>
__device__ __forceinline__ void accumulate_fragment(
    const FragmentA& a, const FragmentX& x, float (&sums)[4]) {
#pragma unroll
    for (int i = 0; i < Values; ++i) {
        constexpr int PerAccumulator = Values > 4 ? 2 : 1;
        sums[i / PerAccumulator] = fmaf(
            as_float(a(i)), as_float(x(i)), sums[i / PerAccumulator]);
    }
}

template <class T, int Group, int Warps, bool Vectorized, bool Split>
__global__ __launch_bounds__(Warps * 32)
void kernel(const T* __restrict__ a, const T* __restrict__ x, T* __restrict__ y,
    float* __restrict__ partials, int m, int k, int splits) {
    using namespace cute;
    static_assert(Group == 4 || Group == 8 || Group == 16 || Group == 32);
    using Config = CopyConfig<T, Group, Warps, Vectorized>;
    constexpr int Rows = Config::kRows;
    constexpr int V = Config::kValues;

    // 使用与 TiledCopy 完全相同的 ThreadLayout 计算输出行和 subgroup lane，
    // 使访存分配、shuffle 的 width 和最终写回的 owner 保持一致。
    auto thread_coord = typename Config::ThreadLayout{}.get_flat_coord(threadIdx.x);
    int row_base = blockIdx.x * Rows;
    int row = row_base + get<0>(thread_coord);
    int lane = get<1>(thread_coord);
    unsigned mask = __ballot_sync(0xffffffffu, row < m);
    if (row >= m) {
        return;  // 整个 subgroup 一起退出，余下 subgroup 的 shuffle mask 仍有效。
    }

    // Split-K 以完整拷贝向量为单位切分，禁止截断一次 128-bit load。
    // scalar fallback 中 V=1，故任意 K 都能被完整覆盖；分区允许为空。
    int vectors = k / V;
    int start = int(std::int64_t(vectors) * blockIdx.y / splits);
    int end = int(std::int64_t(vectors) * (blockIdx.y + 1) / splits);

    auto matrix = make_tensor(
        make_gmem_ptr(a), make_shape(m, k), make_stride(std::int64_t(k), _1{}));
    // x 沿 M 维的 stride 为 0：这是共享同一条向量的逻辑广播视图，
    // 没有生成或存储 M 份 x。实际读取由 cache 复用。
    auto vector = make_tensor(
        make_gmem_ptr(x), make_shape(m, k), make_stride(_0{}, _1{}));
    auto origin = make_coord(row_base, std::int64_t(start) * V);
    auto tiles_a = local_tile(
        domain_offset(origin, matrix), typename Config::Tile{}, make_coord(_0{}, _));
    auto tiles_x = local_tile(
        domain_offset(origin, vector), typename Config::Tile{}, make_coord(_0{}, _));

    typename Config::TiledCopy tiled_copy;
    auto thread_copy = tiled_copy.get_slice(threadIdx.x);
    // 分片形状：(copy_values, copy_rows, copy_columns, K_tiles)。
    // 此处每个线程每轮恰好持有一条向量，copy_rows=copy_columns=1。
    auto source_a = thread_copy.partition_S(tiles_a);
    auto source_x = thread_copy.partition_S(tiles_x);
    auto registers_a = make_tensor<T>(shape(source_a(_, _, _, 0)));
    auto registers_x = make_tensor<T>(shape(source_x(_, _, _, 0)));
    static_assert(size(registers_a) == V && size(registers_x) == V);

    float sums[4] = {0, 0, 0, 0};
    for (std::int64_t base = start, tile = 0; base < end; base += Group, ++tile) {
        // local_tile 不隐式检查边界。最后一个 K tile 的部分 lane 可能超出
        // 当前 split；这些 lane 不执行 load/FMA，但仍参加后面的行内归约。
        if (base + lane < end) {
            copy(tiled_copy, source_a(_, _, _, tile), registers_a);
            copy(tiled_copy, source_x(_, _, _, tile), registers_x);
            accumulate_fragment<T, V>(registers_a, registers_x, sums);
        }
    }
    float sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    // Group 为 2 的幂并整除 warp 大小，所以 shuffle 不跨输出行。
    // 只有 lane 0 得到完整行和；无需 shared memory 或 CTA 级 barrier。
#pragma unroll
    for (int offset = Group / 2; offset; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset, Group);
    }
    if (lane == 0) {
        if constexpr (Split) {
            // 每个 (split,row) 仅有一个 writer；workspace 不需要预先清零。
            auto output = make_tensor(
                make_gmem_ptr(partials), make_shape(splits, m),
                make_stride(std::int64_t(m), _1{}));
            output(blockIdx.y, row) = sum;
        } else {
            auto output = make_tensor(make_gmem_ptr(y), make_shape(m));
            output(row) = from_float<T>(sum);
        }
    }
}

template <class T>
__global__ void reduce(T* y, const float* partials, int m, int splits) {
    using namespace cute;
    auto input = make_tensor(
        make_gmem_ptr(partials), make_shape(splits, m),
        make_stride(std::int64_t(m), _1{}));
    auto output = make_tensor(make_gmem_ptr(y), make_shape(m));
    // partial 按 [split,M] 存储，相邻线程读取相邻行，保证每轮 s 的合并访问。
    // 同一 stream 上的第二次 launch 保证所有 partial 已完成；固定 s 顺序
    // 避免 atomic 更新顺序的不确定性，只在最终写回时转换为输出类型。
    for (std::size_t row = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        row < std::size_t(m); row += std::size_t(blockDim.x) * gridDim.x) {
        float sum = 0;
        for (int s = 0; s < splits; ++s) {
            sum += input(s, row);
        }
        output(row) = from_float<T>(sum);
    }
}

}  // namespace gemv_detail

// 只保存 host 端 launch 所需的 POD 参数，可在计时/Graph capture 前构造并复用。
// workspace 布局为 float[splits][m]；splits=1 时直接写 y，workspace 为 0。
struct GemvPlan {
    int m = 0, k = 0, group = 32, warps = 4, splits = 1;
    std::size_t workspace_bytes = 0;
};

// group=0 根据 K 选择每行 lane 数，splits=0 根据输出 CTA 数估计并行度。
// 默认分区保留至少约 2048 个 K 元素，避免小工作量被第二次 launch 的开销淹没。
// 这些只是启发式；benchmark 可以把实测选出的 group/warps/splits 显式传入。
// 此函数不访问 CUDA runtime，sm_count 应在调用方初始化时查询。
inline cudaError_t make_gemv_plan(GemvPlan* plan, int m, int k, int sm_count = 80,
    int group = 0, int warps = 4, int splits = 0) {
    if (!plan || m < 0 || k < 0 || sm_count <= 0 || splits < 0 ||
        (warps != 4 && warps != 8)) return cudaErrorInvalidValue;
    if (!group) group = k >= 1024 ? 32 : (k >= 256 ? 16 : 8);
    if (group != 4 && group != 8 && group != 16 && group != 32) return cudaErrorInvalidValue;
    int rows = warps * 32 / group;
    int blocks = m / rows + (m % rows != 0);
    if (!splits) {
        std::int64_t need = (std::int64_t(sm_count) * 2 + std::max(1, blocks) - 1) / std::max(1, blocks);
        splits = std::max(1, int(std::min<std::int64_t>({need, 32, k / 2048})));
    }
    // grid.y 用作 split 维，因此要检查 65535 上限；乘法在 size_t 中完成。
    if (splits > 65535 || (k > 0 && splits > k)) return cudaErrorInvalidValue;
    if (!k || !m) splits = 1;
    if (std::size_t(m) > std::numeric_limits<std::size_t>::max() / sizeof(float) / splits)
        return cudaErrorInvalidValue;
    *plan = {m, k, group, warps, splits, splits > 1 ? std::size_t(m) * splits * sizeof(float) : 0};
    return cudaSuccess;
}

namespace gemv_detail {
// 将运行时计划转成有限个编译期实例，使拷贝宽度、线程布局和 shuffle width
// 都能静态展开。保持选择逻辑在 host，避免给 device 热路径增加分支。
template <class T, int Group, int Warps>
inline void dispatch(const GemvPlan& p, const T* a, const T* x, T* y,
    float* workspace, cudaStream_t stream, bool vectorized) {
    constexpr int rows = Warps * 32 / Group;
    dim3 grid(p.m / rows + (p.m % rows != 0), p.splits);
#define CK_GEMV_LAUNCH(V, S) \
    kernel<T, Group, Warps, V, S><<<grid, Warps * 32, 0, stream>>>(a, x, y, workspace, p.m, p.k, p.splits)
    if (p.splits == 1) {
        if (vectorized) { CK_GEMV_LAUNCH(true, false); }
        else { CK_GEMV_LAUNCH(false, false); }
    } else {
        if (vectorized) { CK_GEMV_LAUNCH(true, true); }
        else { CK_GEMV_LAUNCH(false, true); }
    }
#undef CK_GEMV_LAUNCH
}
}  // namespace gemv_detail

template <class T>
inline cudaError_t launch_gemv(const GemvPlan& plan, const T* a, const T* x, T* y,
    void* workspace = nullptr, std::size_t workspace_bytes = 0,
    cudaStream_t stream = nullptr) {
    static_assert(std::is_same_v<T, __half> || std::is_same_v<T, float>, "FP16 or FP32 only");
    // 重新验证公开的 plan，防止调用方修改维度却沿用旧 workspace 大小。
    // M=0 是空操作；K=0 时只清零 y，不要求 A/x 指针存在。
    GemvPlan checked;
    auto status = make_gemv_plan(&checked, plan.m, plan.k, 80, plan.group, plan.warps, plan.splits);
    if (status != cudaSuccess || checked.splits != plan.splits || checked.group != plan.group ||
        checked.workspace_bytes != plan.workspace_bytes) return cudaErrorInvalidValue;
    if (!plan.m) return cudaSuccess;
    if (!y || reinterpret_cast<std::uintptr_t>(y) % alignof(T)) return cudaErrorInvalidValue;
    if (!plan.k) return cudaMemsetAsync(y, 0, std::size_t(plan.m) * sizeof(T), stream);
    if (!a || !x || reinterpret_cast<std::uintptr_t>(a) % alignof(T) ||
        reinterpret_cast<std::uintptr_t>(x) % alignof(T) ||
        (checked.workspace_bytes && (!workspace || workspace_bytes < checked.workspace_bytes ||
                reinterpret_cast<std::uintptr_t>(workspace) % alignof(float))))
        return cudaErrorInvalidValue;
    // 每个矩阵行都要满足 16-byte 对齐：仅检查首地址还不够，K 的元素数
    // 也必须能整除向量宽度。否则使用同一套 CuTe 映射的标量 Copy_Atom。
    bool vectorized = plan.k % (16 / sizeof(T)) == 0 &&
        (reinterpret_cast<std::uintptr_t>(a) | reinterpret_cast<std::uintptr_t>(x)) % 16 == 0;
#define CK_GEMV_GROUP(G) \
    case G: \
            if (plan.warps == 4) gemv_detail::dispatch<T, G, 4>(plan, a, x, y, static_cast<float*>(workspace), stream, vectorized); \
            else gemv_detail::dispatch<T, G, 8>(plan, a, x, y, static_cast<float*>(workspace), stream, vectorized); \
    break
    switch (plan.group) { CK_GEMV_GROUP(4); CK_GEMV_GROUP(8); CK_GEMV_GROUP(16); CK_GEMV_GROUP(32); }
#undef CK_GEMV_GROUP
    status = cudaGetLastError();
    if (status != cudaSuccess || plan.splits == 1) return status;
    // partial 与 reduce 在同一 stream 提交，设备端有序执行，无 host 同步。
    // 调用方必须保持 workspace 有效，直至两次 kernel 均完成。
    unsigned blocks = unsigned(std::min<std::size_t>((std::size_t(plan.m) + 255) / 256, 65535));
    gemv_detail::reduce<<<blocks, 256, 0, stream>>>(y, static_cast<const float*>(workspace), plan.m, plan.splits);
    return cudaGetLastError();
}

}  // namespace cute_kernel
