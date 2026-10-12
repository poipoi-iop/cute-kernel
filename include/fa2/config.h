#pragma once
#include <cuda_runtime.h>

namespace fa2 {
struct FlashAttnConfig {
    const void* __restrict__ q_ptr;
    const void* __restrict__ k_ptr;
    const void* __restrict__ v_ptr;
    void* __restrict__ o_ptr;

    int batch_size;
    int seq_len_q;
    int seq_len_kv;
    int head_dim;
    int num_heads;
    int num_kv_heads;

    int q_batch_stride;
    int q_seq_stride;
    int q_head_stride;

    int k_batch_stride;
    int k_seq_stride;
    int k_head_stride;

    int v_batch_stride;
    int v_seq_stride;
    int v_head_stride;

    int o_batch_stride;
    int o_seq_stride;
    int o_head_stride;

    float scale_softmax;
    float scale_softmax_log2;
};
}
