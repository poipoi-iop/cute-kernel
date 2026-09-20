#include <cuda_runtime.h>


__device__ float warp_reduce_sum(float val) {
    for (int s = 16; s >= 1; s >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, s);
    }
    return val;
}

template <const int ROW, const int COL>
__global__ void rms_norm(float* input, float* output, float eps = 1e-5) {
    unsigned bid = blockIdx.x;
    unsigned tid = threadIdx.x;
    unsigned block_size = blockDim.x;
    unsigned lane_id = tid & 31;
    unsigned warp_id = tid >> 5;

    float* inp = input + bid * COL;
    float* oup = output + bid * COL;
    float4* inp4 = reinterpret_cast<float4*>(inp);
    float4* oup4 = reinterpret_cast<float4*>(oup);
    unsigned n4 = COL / 4;

    __shared__ float warp_result[32];
    __shared__ float final_sum;

    float sum = 0.f;
    for (int i = tid; i < n4; i += block_size) {
        float4 tmp = inp4[i];
        sum += (tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z + tmp.w * tmp.w);
    }

    for (int i = tid + n4 * 4; i < COL; i += block_size) {
        float tmp = inp[i];
        sum += (tmp * tmp);
    }

    if (lane_id == 0) warp_result[warp_id] = sum;
    __syncthreads();

    int num_warps = block_size / 32;
    if (warp_id == 0) {
        sum = (lane_id < num_warps) ? warp_result[lane_id] : 0.f;
        warp_reduce_sum(sum);
    }
    
    if (tid == 0) {
        final_sum = rsqrtf(sum / COL + eps);
    }
    __syncthreads();
    for (int i = tid; i < n4; i += block_size) {
        float4 tmp = inp4[i];
        float4 result;
        result.x = tmp.x * final_sum;
        result.y = tmp.y * final_sum;
        result.z = tmp.z * final_sum;
        result.w = tmp.w * final_sum;
        oup4[i] = result;
    }

    for (int i = tid + n4 * 4; i < COL; i += block_size) {
        oup[i] = inp[i] * final_sum;
    }
}