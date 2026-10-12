#pragma once

#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace fa2 {
using namespace cute;

// 这里这样写是因为 Tensor 是类模板，在 kernel 中可以直接用是因为进行了类型推导
template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4,
          typename TiledMMA, typename TiledCopyA, typename TiledCopyB, typename ThrCopyA, typename ThrCopyB>
CUTLASS_DEVICE void gemm(
    Tensor0& acc_m,  // [MMA, MMA_M, MMA_N]
    Tensor1& tCrA,   // [MMA, MMA_M, MMA_K]
    Tensor2& tCrB,   // [MMA, MMA_N, MMA_K]
    Tensor3 tCsA,   // [MMA, MMA_M, MMA_K]
    Tensor4 tCsB,   // [MMA, MMA_N, MMA_K]
    TiledMMA mma,
    TiledCopyA tiled_copy_a,
    TiledCopyB tiled_copy_b,
    ThrCopyA thr_copy_a,
    ThrCopyB thr_copy_b
) {
    auto tXrA = thr_copy_a.retile_D(tCrA);
    auto tXrB = thr_copy_b.retile_D(tCrB);

    copy(tiled_copy_a, tCsA(_, _, _0{}), tXrA(_, _, _0{}));
    copy(tiled_copy_b, tCsB(_, _, _0{}), tXrB(_, _, _0{}));

    for (int k = 0; k < size<2>(tCrA); ++k) {
        // prefetch
        if (k < size<2>(tCrA) - 1) {
            copy(tiled_copy_a, tCsA(_, _, k + 1), tXrA(_, _, k + 1));
            copy(tiled_copy_b, tCsB(_, _, k + 1), tXrB(_, _, k + 1));
        }
        cute::gemm(mma, tCrA(_, _, k), tCrB(_, _, k), acc_m);
    }
}

template <typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename TiledMma, typename TiledCopy,
          typename ThrCopy>
CUTLASS_DEVICE void
gemm_rs(Tensor0 &acc,        // (MMA, MMA_M, MMA_N)
        Tensor1 &tCrA,       // (MMA, MMA_M, MMA_K)
        Tensor2 &tCrB,       // (MMA, MMA_N, MMA_K)
        Tensor3 const &tCsB, // (CPY, MMA_N, MMA_K)
        TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
        ThrCopy smem_thr_copy_B) {

  Tensor tXrB = smem_thr_copy_B.retile_D(tCrB);
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tXrB(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); i++) {
    // prefetch next block
    if (i < size<2>(tCrB) - 1) {
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tXrB(_, _, i + 1));
    }

    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto
convert_type(Tensor<Engine, Layout> const &tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // HACK: this requires tensor to be "contiguous"
  auto frag =
      convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(
          tensor.data()));
  // Own the converted registers; returning a pointer to frag would dangle.
  auto result = make_tensor<To_type>(tensor.layout());
  *reinterpret_cast<cutlass::Array<To_type, numel> *>(result.data()) = frag;
  return result;
}

template <typename Layout>
__forceinline__ __device__ auto
convert_c_frag_to_a_frag(Layout const &acc_layout) {
  using _ = Underscore;

  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  auto l = logical_divide(acc_layout,
                          Shape<_, _, _2>{}); // (4, MMA_M, (2, MMA_N / 2)))
  return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l),
                     get<2, 1>(l));

}
}
