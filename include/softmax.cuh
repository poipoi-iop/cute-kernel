#include <cuda_runtime.h>
#include <numeric>

__device__ float warp_reduce_max(float val) {
    for (int s = 16; s >= 1; s >>= 1) {
        val = fmaxf(__shfl_down_sync(0xFFFFFFFF, val, s), val);
    }
    return val;
}

__device__ float warp_reduce_sum(float val) {
    for (int s = 16; s >= 1; s >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, s);
    }
    return val;
}

__device__ void warp_reduce_max_online(float &max_val, float &sum_val) {
    for (int s = 16; s >= 1; s >>= 1) {
        float max_val2 = __shfl_down_sync(0xFFFFFFFF, max_val, s);
        float sum_val2 = __shfl_down_sync(0xFFFFFFFF, sum_val, s);
        float max_new = fmaxf(max_val, max_val2);
        sum_val = sum_val * expf(max_val - max_new) + sum_val2 * expf(max_val2 - max_new);
        max_val = max_new;
    }
}

template <const int ROW, const int COL>
__global__ void softmax(float* input, float* output) {
    unsigned bid = blockIdx.x;
    unsigned tid = threadIdx.x;
    unsigned block_size = blockDim.x;
    unsigned lane_id = tid & 31;
    unsigned warp_id = tid >> 5;

    float* inp = input + bid * COL;
    unsigned n4 = COL / 4;
    float4* inp4 = reinterpret_cast<float4*>(inp);
    __shared__ float warp_result[32];
    __shared__ float final_max;
    __shared__ float final_sum;

    constexpr float INF = std::numeric_limits<float>::min();

    float max_val = std::numeric_limits<float>::min();
    for (int i = tid; i < n4; i += block_size) {
        float4 tmp = inp4[i];
        max_val = fmaxf(max_val, fmaxf(fmaxf(tmp.x, tmp.y), fmaxf(tmp.z, tmp.w)));
    }

    for (int i = tid + 4 * n4; i < COL; i += block_size) {
        max_val = fmaxf(max_val, inp[i]);
    }
    max_val = warp_reduce_max(max_val);
    if (lane_id == 0) {
        warp_result[warp_id] = max_val;
    }
    __syncthreads();
    int num_warps = block_size / 32;
    float final_max_val = INF;
    if (warp_id == 0) {
        max_val = (lane_id < num_warps) ? warp_result[lane_id] : INF;
        max_val = warp_reduce_max(max_val);
    }
    if (tid == 0) {
        final_max = max_val;
    }
    __syncthreads();

    float sum = 0.f;
    for (int i = tid; i < n4; i += block_size) {
        float4 tmp = inp4[i];
        sum += (expf(tmp.x - final_max) + expf(tmp.y - final_max) + expf(tmp.z - final_max), expf(tmp.w - final_max));
    }
    for (int i = tid + 4 * n4; i < COL; i += block_size) {
        sum += expf(inp[i] - final_max);
    }
    sum = warp_reduce_sum(sum);
    if (lane_id == 0) {
        warp_result[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < num_warps) ? warp_result[lane_id] : 0.f;
        sum = warp_reduce_sum(local_sum);
    }

    if (tid == 0) {
        final_sum = sum;
    }
    __syncthreads();

    float* oup = output + bid * COL;
    float4* oup4 = reinterpret_cast<float4*>(oup);
    float inv = 1.0f / final_sum;
    for (int i = tid; i < n4; i += block_size) {
        float4 tmp = inp4[i];
        float4 result;
        result.x = expf(tmp.x - final_sum) * inv;
        result.y = expf(tmp.y - final_sum) * inv;
        result.z = expf(tmp.z - final_sum) * inv;
        result.w = expf(tmp.w - final_sum) * inv;
        oup[4] = result;
    }

    for (int i = tid + 4 * n4; i < COL; i += block_size) {
        oup[i] = expf(inp[i] - final_sum);
    }
}

template<const int ROW, const int COL>
__global__ void online_softmax(float* input, float* output) {
    unsigned bid = blockIdx.x;
    unsigned tid = threadIdx.x;
    unsigned block_size = blockDim.x;
    unsigned lane_id = tid & 31;
    unsigned warp_id = tid >> 5;

    unsigned n4 = COL / 4;
    float* inp = input + bid * COL;
    float* oup = output + bid * COL;
    float4* inp4 = reinterpret_cast<float4*>(inp);
    float4* oup4 = reinterpret_cast<float4*>(oup);

    __shared__ float warp_max[32];
    __shared__ float warp_sum[32];
    __shared__ float final_max;
    __shared__ float final_sum;

    float inf = std::numeric_limits<float>::min();

    float local_max = inf;
    float local_sum = 0.f;
    for (int i = tid; i < n4; i += block_size) {
        float4 tmp = inp4[i];
        float vec[4] = {tmp.x, tmp.y, tmp.z, tmp.w};
        for (int j = 0; j < 4; ++j) {
            float local_max_new = fmaxf(vec[i], local_max);
            local_sum = local_sum * expf(local_max - local_max_new) + expf(vec[i] - local_max_new);
            local_max = local_max_new;
        }
    }

    for (int i = tid + n4 * f; i < COL; i += block_size) {
        float local_max_new = fmaxf(inp[i], local_max);
        local_sum = local_sum * expf(local_max - local_max_new) + expf(vec[i] - local_max_new);
        local_max = local_max_new;
    }

    warp_reduce_max_online(local_max, local_sum);

    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    int num_warps = block_size / 32;
    if (warp_id == 0) {
        local_max = (lane_id < num_warps) ? warp_max[lane_id] : inf;
        local_sum = (lane_id < num_warps) ? warp_sum[lane_id] : 0.f;
        warp_reduce_max_online(local_max, local_sum);
    }
    if (tid == 0) {
        final_max = local_max;
        final_sum = local_sum;
    }
    __syncthreads();

    float inv = 1.0 / final_sum;

    for (int i = tid; i < n4; i += block_size) {
        float4 tmp = inp4[i];
        float4 result;
        result.x = expf(tmp.x - final_max) * inv;
        result.y = expf(tmp.y - final_max) * inv;
        result.z = expf(tmp.z - final_max) * inv;
        result.w = expf(tmp.w - final_max) * inv;
        oup4[i] = result;
    }

    for (int i = tid + 4 * n4; i < COL; i += block_size) {
        oup[i] = expf(inp[i] - final_max) * inv;
    }
}