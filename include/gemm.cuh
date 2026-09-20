#include <cute/layout.hpp>
#include <cute/tensor.hpp>

namespace config {
using namespace cute;
template <typename T_, const int kTileM_ = 128, const int kTIleN_ = 128, const int kTileK_ = 32,
          const int kStage_ = 5, const int kSmemLayoutCBatch_ = 2, 
          typename ComputeType = T_>
class Config {
    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTIleN_;
    static constexpr int kTileK = kTileK_;
    static constexpr int kStage = kStage_;
    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;
    static constexpr int kMmaVRepeatM = 1;
    static constexpr int kMmaVRepeatN = 2;
    static constexpr int kMmaVRepeatK = 1;
    static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;

    using T = T_;

    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    using mma_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_shape{});

    using TiledMMA = decltype(
        mma_op{},
        make_layout(make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})),
        Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>
    );

    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA = decltype(
        make_tiled_copy(
            g2s_copy_atom{},
            make_layout(make_shape(_32{}, _4{}), make_stride(_4{}, _1{})),
            make_layout(make_shape(_1{}, _8{}))
        )
    );
    using G2SCopyB = G2SCopyA;

    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    static constexpr int kCopySwizzleB = 3;
    static constexpr int kCopySwizzleM = 3;
    static constexpr int kCopySwizzleS = 3;

    using SmemLayoutAtom = decltype(composition(
        Swizzle<Int<kCopySwizzleB>, Int<kCopySwizzleM>, Int<kCopySwizzleS>>,
        make_layout(make_shape(_8{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, _1{}))
    ));

    using SmemLayoutA = decltype(
        tile_to_shape(SmemLayoutAtom{}, make_shape(Int<kTileM>{}, Int<kTileN>{}, Int<kStage>{}))
    );
    using SmemLayoutB = decltype(
        tile_to_shape(SmemLayoutAtom{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{}))
    );

    using SmemLayoutCAtom = decltype(composition(
        Swizzle<2, 3, 3>{},
        make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}), make_stride(Int<KMmaPN>{}, _1{}));
    ));

    using SmemLayoutC = decltype(
        tile_to_shape(SmemLayoutCAtom{}, make_shape(Int<kMmaPM>{}, Int<KMmaPN>{}, Int<kSmemLayoutCBatch>))
    );

    using r2s_copy_atom_c = Copy_Atom<UniversalCopy<int>, T>;
    using s2g_copy_atom_c = Copy_Atom<UniversalCopy<uint128_t>, T>;
    using S2GCopyC = decltype(
        make_tiled_copy(s2g_copy_atom_c{}, make_layout(make_shape(_32{}, _4{}), make_stride(_4{}, _1{})), make_layout(_1{}, _4{}))
    );

    static constexpr int kThreadNum = size(TiledMMA{});
    static constexpr int shm_size_AB = cosize(SmemLayoutA{}) + cosize(SmemLayoutB{});
    static constexpr int shm_size_C = cosize(SmemLayoutC{});
    static constexpr int kShmSize = cute::max(shm_size_AB, shm_size_C) * sizeof(T);
};

}

template <typename Config>
__global__ void hgemm(void* C_ptr, const void* A_ptr, const void* B_ptr, int M, int N, int K) {
    using namespace cute;

    using T = typename Config::T;
    using TiledMMA = typename Config::TiledMMA;
    using G2SCopyA = typename Config::G2SCopyA;
    using G2SCopyB = typename Config::G2SCopyB;
    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;
    using SmemLayoutA = typename Config::SmemLayoutA;
    using SmemLayoutB = typename Config::SmemLayoutB;
    using SmemLayoutC = typename Config::SmemLayoutC;
    using R2SCopyAtomC = typename Config::r2s_copy_atom_c;
    using S2GCopyC = typename Config::S2GCopyC;

    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;
    constexpr int kStage = Config::kStage;

    Tensor a = make_tensor(make_gmem_ptr((T*)A_ptr), make_shape(Int<M>{}, Int<K>{}), make_stride(Int<K>{}, _1{}));
    Tensor b = make_tensor(make_gmem_ptr((T*)B_ptr), make_shape(Int<N>{}, Int<K>{}), make_stride(Int<K>{}, _1{}));
    Tensor c = make_tensor(make_gmem_ptr((T*)C_ptr), make_shape(Int<M>{}, Int<N>{}), make_stride(Int<N>{}, _1{}));

    unsigned ix = blockIdx.x;
    unsigned iy = blockIdx.y;

    // ga: (kTileM, kTileK, num_k_tiles)
    Tensor ga = local_tile(a, make_shape(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    // gb: (kTileN, kTileK, num_k_tiles)
    Tensor gb = local_tile(b, make_shape(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    // gc: (kTileM, kTileN)
    Tensor gc = local_tile(c, make_shape(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));

    unsigned tid = threadIdx.x;

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);
    // tra: (MMA, MMA_M, MMA_K)
    Tensor tra = thr_mma.partition_fragment_A(ga(_, _, 0));
    // trb: (MMA, MMA_N, MMA_K)
    Tensor trb = thr_mma.partition_fragment_B(gb(_, _, 0));
    // tra: (MMA, MMA_M, MMA_N)
    Tensor trc = thr_mma.partition_fragment_C(gc(_, _));
    clear(trc);

    extern __shared__ T shm_raw[];
    T* a_smem = shm_raw;
    T* b_smem = a_smem + cosize(SmemLayoutA{});
    Tensor sa = make_tensor(make_smem_ptr(a_smem), SmemLayoutA{});
    Tensor sb = make_tensor(make_smem_ptr(b_smem), SmemLayoutB{});

    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto thr_copy_s2r_a = s2r_tiled_copy_a.get_slice(tid);
    // tsra: (CPY, CPY_M, CPY_K, k_stage)
    Tensor tsra = thr_copy_s2r_a.partition_S(sa);
    // tsra_reg: (CPY, CPY_M, CPY_K)
    Tensor tsra_reg = thr_copy_s2r_a.retile_D(tra); 

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto thr_copy_s2r_b = s2r_tiled_copy_b.get_slice(tid);
    // tsra: (CPY, CPY_N, CPY_K, k_stage)
    Tensor tsrb = thr_copy_s2r_b.partition_S(sb);
    // tsra_reg: (CPY, CPY_N, CPY_K)
    Tensor tsrb_reg = thr_copy_s2r_b.retile_D(trb);

    G2SCopyA g2s_tiled_copy_a;
    auto thr_copy_g2s_a = g2s_tiled_copy_a.get_slice(tid);
    // tgra: (CPY, CPY_M, CPY_K, k)
    Tensor tgra = thr_copy_g2s_a.partition_S(ga);
    // tgra_smem: (CPY, CPY_M, CPY_K, k_stage)
    Tensor tgra_smem = thr_copy_g2s_a.partition_D(sa);

    G2SCopyB g2s_tiled_copy_b;
    auto thr_copy_g2s_b = g2s_tiled_copy_b.get_slice(tid);
    // tgrb: (CPY, CPY_N, CPY_K, k)
    Tensor tgrb = thr_copy_g2s_b.partition_S(gb);
    // tgrb_smem: (CPY, CPY_N, CPY_K, k_stage)
    Tensor tgrb_smem = thr_copy_g2s_b.partition_D(sb);

    int tile_to_read = 0;
    int smem_to_read = 0;
    int smem_to_write = 0;
    for (int s = 0; s < kStage; ++s) {
        copy(g2s_tiled_copy_a, tgra(_, _, _, s), tgra_smem(_, _, _, s));
        copy(g2s_tiled_copy_b, tgrb(_, _, _, s), tgrb_smem(_, _, _, s));
        cp_async_fence();

        ++tile_to_read;
        ++smem_to_write;
    }

    cp_async_wait(kStage - 2);
    __syncthreads();

    int ik = 0;
    copy(s2r_tiled_copy_a, tsra(_, _, ik, smem_to_read), tsra_reg(_, _, ik));
    copy(s2r_tiled_copy_b, tsrb(_, _, ik, smem_to_read), tsrb_reg(_, _, ik));

    int k_tiles = K / kTileK;
    for (int i_tile = 0; i_tile < k_tiles; ++i_tile) {
        int nk = size<2>(tra);
        for (int ik = 0; ik < nk; ++ik) {
            int ik_next = (ik + 1) % nk;
            if (ik == nk - 1) {
                cp_async_wait(kStage - 2);
                __syncthreads();

                smem_to_read = (smem_to_read + 1) % kStage;
            }

            copy(s2r_tiled_copy_a, tsra(_, _, ik_next, smem_to_read), tsra_reg(_, _, ik));
            copy(s2r_tiled_copy_b, tsrb(_, _, ik_next, smem_to_read), tsrb_reg(_, _, ik));

            if (ik == 0) {
                if (tile_to_read < n_tile) {
                    copy(g2s_tiled_copy_a, tgra(_, _, _, tile_to_read), tgra_smem(_, _, _, smem_to_write));
                    copy(g2s_tiled_copy_b, tgrb(_, _, _, tile_to_read), tgrb_smem(_, _, _, smem_to_write));

                    ++tile_to_read;
                    smem_to_write = (smem_to_write + 1) % kStage;
                }
                cp_async_fence();
            }

            gemm(tiled_mma, trc, tra(_, _, ik), trb(_, _, ik), trc);
        }
    }

    auto sc = make_tensor(sa(_, _, smem_to_read).data(), SmemLayoutC{});
    auto r2s_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_copy_c.get_slice(tid);
    Tensor r2s_r = r2s_thr_copy_c.retile_S(trc);
    Tensor r2s_s = r2s_thr_copy_c.partition_D(sc);

    S2GCopyC s2g_copy_c;
    auto s2g_thr_copy_c = s2g_copy_c.get_thread_slice(tid);
    Tensor s2g_s = s2g_thr_copy_c.partition_S(sc);
    Tensor s2g_g = s2g_thr_copy_c.partition_D(gc);

    auto r2s_sx = group_modes<1, 3>(r2s_s);
    auto s2g_gx = group_modes<1, 3>(s2g_g);

    int step = size<3>(r2s_r);
    #pragma unroll
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
    // reg -> shm
#pragma unroll
    for (int j = 0; j < step; ++j) {
        // we add a temp tensor to cope with accumulator and output data type
        // difference
        auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
        cute::copy(tCrC_r2sx(_, i + j), t);

        cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads();

#pragma unroll
    // shm -> global
    for (int j = 0; j < step; ++j) {
        cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }

    __syncthreads();
  }
}
