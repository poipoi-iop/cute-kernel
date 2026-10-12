#pragma once

// GEMV: y[M] = A[M,K] * x[K]，A 为连续行主序，T = __half 或 float，FP32 累加。
// 实现思路与实测数据见 docs/gemv.md。
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace cute_kernel {

// 16 字节 = 8 个 half 或 4 个 float 的点积，累加到 FP32。
__device__ __forceinline__ float dot16(uint4 a, uint4 b, float acc, __half) {
    const __half2* a2 = reinterpret_cast<const __half2*>(&a);
    const __half2* b2 = reinterpret_cast<const __half2*>(&b);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 fa = __half22float2(a2[i]), fb = __half22float2(b2[i]);
        acc = fmaf(fa.x, fb.x, acc);
        acc = fmaf(fa.y, fb.y, acc);
    }
    return acc;
}
__device__ __forceinline__ float dot16(uint4 a, uint4 b, float acc, float) {
    acc = fmaf(__uint_as_float(a.x), __uint_as_float(b.x), acc);
    acc = fmaf(__uint_as_float(a.y), __uint_as_float(b.y), acc);
    acc = fmaf(__uint_as_float(a.z), __uint_as_float(b.z), acc);
    acc = fmaf(__uint_as_float(a.w), __uint_as_float(b.w), acc);
    return acc;
}

// 每 G 个相邻线程负责一行，一个 CTA 处理 THREADS / G 行。
//   G <= 32 : 一个 warp 内放 32/G 行，只用 shuffle 归约。
//   G >  32 : 一个 CTA 只算一行（G == THREADS），warp 间再经 shared memory 归约。
// VEC=true 时每次加载 16 字节（要求 A、x 16 字节对齐且 K 是 16/sizeof(T) 的倍数），
// 否则逐元素加载。每个线程同时发出 UNROLL 个独立的 load，提高在途访存量。
//
// Split-K（仅 G > 32）：gridDim.y 个 CTA 各算同一行的一段 K，把部分和写进
// partial[split][M]；最后一个到达的 CTA（由 counter[row] 原子计数判定）按固定
// 顺序求和并写 y，然后把 counter 清零，供下一次调用复用。
template <class T, int G, int THREADS, bool VEC>
__global__ void __launch_bounds__(THREADS)
gemv_kernel(const T* __restrict__ A, const T* __restrict__ x, T* __restrict__ y, int M, int K,
            float* __restrict__ partial, unsigned* __restrict__ counter) {
    static_assert(G <= 32 || G == THREADS, "G > 32 means one row per CTA");
    constexpr int ROWS = THREADS / G;
    constexpr int UNROLL = 4;

    const int lane = threadIdx.x % G;
    const int row = blockIdx.x * ROWS + threadIdx.x / G;
    const T* a = A + std::size_t(min(row, M - 1)) * K;

    const int n = VEC ? K / (16 / sizeof(T)) : K;
    const int chunk = (n + gridDim.y - 1) / gridDim.y;
    const int beg = min(n, int(blockIdx.y) * chunk);
    const int end = min(n, beg + chunk);

    float acc[UNROLL] = {};
    int i = beg + lane;
    if constexpr (VEC) {
        const uint4* a4 = reinterpret_cast<const uint4*>(a);
        const uint4* x4 = reinterpret_cast<const uint4*>(x);
        for (; i + (UNROLL - 1) * G < end; i += UNROLL * G) {
            uint4 va[UNROLL], vx[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                va[u] = __ldg(a4 + i + u * G);
                vx[u] = __ldg(x4 + i + u * G);
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) acc[u] = dot16(va[u], vx[u], acc[u], T());
        }
        for (; i < end; i += G) acc[0] = dot16(__ldg(a4 + i), __ldg(x4 + i), acc[0], T());
    } else {
        for (; i + (UNROLL - 1) * G < end; i += UNROLL * G) {
            T va[UNROLL], vx[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                va[u] = __ldg(a + i + u * G);
                vx[u] = __ldg(x + i + u * G);
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) acc[u] = fmaf(float(va[u]), float(vx[u]), acc[u]);
        }
        for (; i < end; i += G) acc[0] = fmaf(float(__ldg(a + i)), float(__ldg(x + i)), acc[0]);
    }

    float sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    // G 是 2 的幂，xor 只在同一行的 G 个 lane 之间交换。
#pragma unroll
    for (int offset = (G < 32 ? G : 32) / 2; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);

    if constexpr (G <= 32) {
        if (lane == 0 && row < M) y[row] = T(sum);
    } else {
        constexpr int WARPS = THREADS / 32;
        __shared__ float warp_sum[WARPS];
        if (threadIdx.x % 32 == 0) warp_sum[threadIdx.x / 32] = sum;
        __syncthreads();
        if (threadIdx.x != 0) return;
        sum = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) sum += warp_sum[w];

        if (gridDim.y == 1) {
            y[row] = T(sum);
            return;
        }
        partial[std::size_t(blockIdx.y) * M + row] = sum;
        __threadfence();  // 先让 partial 对其他 CTA 可见，再计数
        if (atomicAdd(counter + row, 1u) != gridDim.y - 1) return;
        __threadfence();
        float total = 0.f;
        for (int s = 0; s < int(gridDim.y); ++s) total += __ldcg(partial + std::size_t(s) * M + row);
        y[row] = T(total);
        counter[row] = 0;
    }
}

template <class T, int G, int THREADS>
inline void gemv_launch(const T* A, const T* x, T* y, int M, int K, bool vec, cudaStream_t stream,
                        int splits = 1, float* partial = nullptr, unsigned* counter = nullptr) {
    constexpr int ROWS = THREADS / G;
    const dim3 grid((M + ROWS - 1) / ROWS, splits);
    if (vec) gemv_kernel<T, G, THREADS, true><<<grid, THREADS, 0, stream>>>(A, x, y, M, K, partial, counter);
    else gemv_kernel<T, G, THREADS, false><<<grid, THREADS, 0, stream>>>(A, x, y, M, K, partial, counter);
}

constexpr int kGemvMaxSplits = 16;

// Split-K 所需 workspace：partial[16][M] float + counter[M] unsigned。
// 分配后清零一次即可（counter 由 kernel 自行归零）；同一时间只能被一个 stream 使用。
inline std::size_t gemv_workspace_bytes(int M) {
    return M > 0 ? std::size_t(M) * (kGemvMaxSplits * sizeof(float) + sizeof(unsigned)) : 0;
}

// y = A * x。workspace 可为空；非空时（gemv_workspace_bytes(M) 字节、已清零），
// 行数少且单行很长的形状会走 Split-K。
//
// 选择每行线程数 G：让每个线程在一行（或一个 split）里分到约 UNROLL=4 次
// 加载，正好填满一轮 unroll；G 上限 512。行多时这样自然得到足够多的
// 线程；行少时 G 变大，一个 CTA 算一行。
template <class T>
inline cudaError_t gemv(const T* A, const T* x, T* y, int M, int K,
                        cudaStream_t stream = nullptr, void* workspace = nullptr) {
    static_assert(sizeof(T) == 2 || sizeof(T) == 4, "__half or float");
    if (M < 0 || K < 0) return cudaErrorInvalidValue;
    if (M == 0) return cudaSuccess;
    if (K == 0) return cudaMemsetAsync(y, 0, std::size_t(M) * sizeof(T), stream);

    const bool vec = K % (16 / sizeof(T)) == 0 &&
        (reinterpret_cast<std::uintptr_t>(A) | reinterpret_cast<std::uintptr_t>(x)) % 16 == 0;
    const std::int64_t row_bytes = std::int64_t(K) * sizeof(T);

    // Split-K：只有行数填不满 SM、且单行 >= 64 KB 时，多出来的并行度才抵得过
    // 跨 CTA 归约的固定开销（约 0.6 us）。每个 split 约 8 KB。
    int splits = 1;
    if (workspace && vec && row_bytes >= 64 * 1024) {
        int device = 0, sms = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        if (M < sms) splits = int(row_bytes / (8 * 1024) < kGemvMaxSplits ? row_bytes / (8 * 1024) : kGemvMaxSplits);
    }
    float* partial = static_cast<float*>(workspace);
    unsigned* counter = workspace ? reinterpret_cast<unsigned*>(partial + std::size_t(kGemvMaxSplits) * M) : nullptr;

    // 一行（或一个 split）的加载次数：向量路径按 16 字节块计。标量路径每次只读
    // 2~4 字节，实测每线程分到约 8 个元素最好，因此按 K/2 计。
    const std::int64_t loads = (vec ? row_bytes / 16 : K / 2) / splits;
    int g = 4;
    while (g < 512 && g * 8 <= loads) g *= 2;

    switch (g) {
        case 4:   gemv_launch<T, 4, 256>(A, x, y, M, K, vec, stream); break;
        case 8:   gemv_launch<T, 8, 256>(A, x, y, M, K, vec, stream); break;
        case 16:  gemv_launch<T, 16, 256>(A, x, y, M, K, vec, stream); break;
        case 32:  gemv_launch<T, 32, 256>(A, x, y, M, K, vec, stream); break;
        case 64:  gemv_launch<T, 64, 64>(A, x, y, M, K, vec, stream, splits, partial, counter); break;
        case 128: gemv_launch<T, 128, 128>(A, x, y, M, K, vec, stream, splits, partial, counter); break;
        case 256: gemv_launch<T, 256, 256>(A, x, y, M, K, vec, stream, splits, partial, counter); break;
        default:  gemv_launch<T, 512, 512>(A, x, y, M, K, vec, stream, splits, partial, counter); break;
    }
    return cudaGetLastError();
}

}  // namespace cute_kernel
