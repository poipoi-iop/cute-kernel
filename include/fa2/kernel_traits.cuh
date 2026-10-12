#pragma once
#include <cute/tensor.hpp>
#include <type_traits>

namespace fa2 {
using namespace cute;

template <const int kTileM_, const int kTileN_, const int kHeadDim_, const int kNWarps_, const int kStages_, typename T_ = half_t>
class Config {
public:
    using T = T_;
    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kStages = kStages_;
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;
    static constexpr int kBlockSmem = (kHeadDim % 64 == 0) ? 64 : 32;
    static constexpr int kSwizzle = (kBlockSmem == 64) ? 3 : 2;

    static constexpr int half_per_4B = sizeof(uint128_t) / sizeof(half_t);  // 128 / 16 = 8
    static constexpr int threads_per_row = kBlockSmem / half_per_4B;
    static constexpr int num_threads_col = threads_per_row;
    static constexpr int num_threads_row = kNThreads / num_threads_col;


    using s2r_copy_traits = Copy_Traits<SM75_U32x4_LDSM_N>;
    using SmemCopyAtom = Copy_Atom<s2r_copy_traits, T>;
    using SmemCopyAtomT = Copy_Atom<Copy_Traits<SM75_U16x8_LDSM_T>, T>;
    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, half_t>;

    static constexpr int kMmaAtomRepeatM = kNWarps;
    static constexpr int kMmaAtomRepeatN = 1;
    static constexpr int kMmaAtomRepeatK = 1;
    static constexpr int kMmaThrRepeatM = kNWarps;
    static constexpr int kMmaThrRepeatN = 2;
    static constexpr int kMmaThrRepeatK = 1;

    using mma_op = std::conditional_t<std::is_same_v<T, half_t>, SM80_16x8x16_F32F16F16F32_TN, SM80_16x8x16_F32BF16BF16F32_TN>;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using mma_shape = typename mma_traits::Shape_MNK;
    using MMA = decltype(
        make_tiled_mma(
            mma_atom{},
            make_layout(
                make_shape(Int<kMmaAtomRepeatM>{}, Int<kMmaAtomRepeatN>{}, Int<kMmaAtomRepeatK>{})
            ),
            make_shape(
                Int<kMmaThrRepeatM * get<0>(mma_shape{})>{},
                Int<kMmaThrRepeatN * get<1>(mma_shape{})>{},
                Int<kMmaThrRepeatK * get<2>(mma_shape{})>{}
            )

        )
    );

    using SmemLayoutQKVAtom = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        make_layout(make_shape(_8{}, Int<kBlockSmem>{}), make_stride(Int<kBlockSmem>{}, _1{}))
    ));

    using SmemLayoutQ = decltype(
        tile_to_shape(SmemLayoutQKVAtom{}, make_shape(Int<kTileM>{}, Int<kHeadDim>{}))
    );
    using SmemLayoutKV = decltype(
        tile_to_shape(SmemLayoutQKVAtom{}, make_shape(Int<kTileN>{}, Int<kHeadDim>{}))
    );

    using SmemLayoutVt = decltype(
        composition(
            SmemLayoutKV{},
            // composition maps this row-major index back to (N, D) coordinates.
            make_layout(make_shape(Int<kHeadDim>{}, Int<kTileN>{}), make_stride(Int<kTileN>{}, _1{}))
        )
    );

    using SmemLayoutVtNoSwizzle = decltype(
        get_nonswizzle_portion(SmemLayoutVt{})
    );

    using SmemLayoutO = decltype(
        tile_to_shape(SmemLayoutQKVAtom{}, make_shape(Int<kTileM>{}, Int<kHeadDim>{}))
    );


    using gmem_copy_traits = Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>;
    using gmem_copy_atom = Copy_Atom<gmem_copy_traits, T>;
    using GmemCopyQKV = decltype(
        make_tiled_copy(gmem_copy_atom{},
            make_layout(make_shape(Int<num_threads_row>{}, Int<num_threads_col>{}), make_stride(Int<num_threads_col>{}, _1{})),
            make_layout(make_shape(_1{}, Int<half_per_4B>{}))
        )
    );
    using GmemCopyO = decltype(
        make_tiled_copy(
            SmemCopyAtomO{},
            make_layout(make_shape(Int<num_threads_row>{}, Int<num_threads_col>{}), make_stride(Int<num_threads_col>{}, _1{})),
            make_layout(make_shape(_1{}, Int<half_per_4B>{}))
        )
    );

    static constexpr int kSmemSize = sizeof(half_t) * (kTileM * kHeadDim + 2 * kTileN * kHeadDim) * kStages;
};
}
