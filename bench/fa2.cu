#include "fa2/flash_attn.cuh"
#include <cmath>

// A small C ABI lets Python use the same CUDA allocations and stream as flash_attn.
// Strides are in elements, ordered (batch, sequence, head) for Q, K, V, O.
template <int HeadDim>
cudaError_t launch(fa2::FlashAttnConfig const& config, cudaStream_t stream, bool setup) {
    using Tile = fa2::Config<64, 64, HeadDim, 4, 1>;
    if (setup) {
        return cudaFuncSetAttribute(fa2::flash_attn_fwd_kernel<Tile>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Tile::kSmemSize);
    }
    fa2::flash_attn_fwd_kernel<Tile><<<
        dim3(config.seq_len_q / Tile::kTileM, config.batch_size * config.num_heads),
        Tile::kNThreads, Tile::kSmemSize, stream>>>(config);
    return cudaGetLastError();
}

extern "C" int fa2_setup(int head_dim) {
    fa2::FlashAttnConfig config{};
    switch (head_dim) {
        case 32: return launch<32>(config, nullptr, true);
        case 64: return launch<64>(config, nullptr, true);
        case 128: return launch<128>(config, nullptr, true);
        default: return cudaErrorNotSupported;
    }
}

extern "C" int fa2_launch(const void* q, const void* k, const void* v, void* o,
    int batch, int seq_q, int seq_kv, int heads, int head_dim,
    const int* strides, cudaStream_t stream) {
    // The current kernel has no tail masks, causal mask, dropout or GQA.
    if (batch <= 0 || heads <= 0 || seq_q <= 0 || seq_kv <= 0 ||
        seq_q % 64 || seq_kv % 64 || !q || !k || !v || !o || !strides)
        return cudaErrorInvalidValue;
    fa2::FlashAttnConfig config{};
    config.q_ptr = q;
    config.k_ptr = k;
    config.v_ptr = v;
    config.o_ptr = o;
    config.batch_size = batch;
    config.seq_len_q = seq_q;
    config.seq_len_kv = seq_kv;
    config.head_dim = head_dim;
    config.num_heads = config.num_kv_heads = heads;
    config.q_batch_stride = strides[0];
    config.q_seq_stride = strides[1];
    config.q_head_stride = strides[2];
    config.k_batch_stride = strides[3];
    config.k_seq_stride = strides[4];
    config.k_head_stride = strides[5];
    config.v_batch_stride = strides[6];
    config.v_seq_stride = strides[7];
    config.v_head_stride = strides[8];
    config.o_batch_stride = strides[9];
    config.o_seq_stride = strides[10];
    config.o_head_stride = strides[11];
    config.scale_softmax = 1.f / std::sqrt(float(head_dim));
    config.scale_softmax_log2 = config.scale_softmax * std::log2(std::exp(1.f));
    switch (head_dim) {
        case 32: return launch<32>(config, stream, false);
        case 64: return launch<64>(config, stream, false);
        case 128: return launch<128>(config, stream, false);
        default: return cudaErrorNotSupported;
    }
}
