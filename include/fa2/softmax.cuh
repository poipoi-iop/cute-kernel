#pragma once

#include "cute/pointer_flagged.hpp"
#include "cute/tensor_impl.hpp"
#include "cutlass/detail/helper_macros.hpp"
#include <cute/tensor.hpp>

namespace fa2 {
using namespace cute;

struct MaxOp {
    CUTLASS_DEVICE float operator() (float a, float b) const {
        return fmaxf(a, b);
    }
};

struct SumOp {
    CUTLASS_DEVICE float operator() (float a, float b) const {
        return a + b;
    }
};

template <bool zero_init, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Op>
CUTLASS_DEVICE void thread_reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& dst, Op op) {
#pragma unroll
    for (int row = 0; row < size<0>(tensor); ++row) {
        dst(row) = zero_init ? tensor(row, 0) : op(tensor(row, 0), dst(row));
#pragma unroll
        for (int col = 1; col < size<1>(tensor); ++col) {
            dst(row) = op(tensor(row, col), dst(row));
        }
        
    }
}

template <int N, typename T, typename Op>
CUTLASS_DEVICE T allreduce(T x, Op op) {
    for (int s = N / 2; s >= 1; s /= 2) {
        x = op(x, __shfl_xor_sync(0xFFFFFFFF, x, s));
    }
    return x;
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Op>
CUTLASS_DEVICE void quad_allreduce_(Tensor<Engine0, Layout0>& dst, Tensor<Engine1, Layout1>& src, Op op) {
    for (int row = 0; row < size<0>(src); ++row) {
        dst(row) = allreduce<4>(src(row), op);
    }
}

template <bool zero_init, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
CUTLASS_DEVICE void reduce_sum(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1>& sum) {
    thread_reduce_<zero_init>(tensor, sum, SumOp{});
}

template <bool zero_init, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
CUTLASS_DEVICE void reduce_max(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1>& max) {
    thread_reduce_<zero_init>(tensor, max, MaxOp{});
    quad_allreduce_(max, max, MaxOp{});
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
CUTLASS_DEVICE void scale_apply_exp2(Tensor<Engine0, Layout0>& tensor, Tensor<Engine1, Layout1>& max, const float softmax_scale_log2) {
#pragma unroll
    for (int row = 0; row < size<0>(tensor); ++row) {
        float adj_max = max(row) * softmax_scale_log2;
#pragma unroll
        for (int col = 0; col < size<1>(tensor); ++col) {
            tensor(row, col) = exp2(tensor(row, col) * softmax_scale_log2 - adj_max);
        }
    }
}

template <typename Layout>
CUTLASS_DEVICE auto convert_layout_rowcol(Layout const& in) {
    auto a_i = logical_divide(in, Shape<_2>{});
    return make_layout(
        make_layout(get<0, 1>(a_i), get<1>(a_i)),
        make_layout(get<0, 0>(a_i), get<2>(a_i))
    );
}

template <int kNRows> struct Softmax {
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));

    TensorT row_max;    // per-row max (m)
    TensorT row_sum;    // per-row sum (l)

    template <bool is_first, typename Tensor0, typename Tensor1>
    CUTLASS_DEVICE void softmax_rescale_o(Tensor0& acc_s, Tensor1& acc_o, float softmax_scale_log2) {
        Tensor scores = make_tensor(acc_s.data(), convert_layout_rowcol(acc_s.layout()));
        if (is_first) {
            reduce_max<true>(scores, row_max);
            scale_apply_exp2(scores, row_max, softmax_scale_log2);
            reduce_sum<true>(scores, row_sum);
        } else {
            Tensor row_max_old = make_fragment_like(row_max);
            copy(row_max, row_max_old);
            reduce_max<false>(scores, row_max);
            Tensor output = make_tensor(acc_o.data(), convert_layout_rowcol(acc_o.layout()));

            for (int r = 0; r < size<0>(output); ++r) {
                float correction = exp2f((row_max_old(r) - row_max(r)) * softmax_scale_log2);
                row_sum(r) *= correction;
                for (int c = 0; c < size<1>(output); ++c) {
                    output(r, c) *= correction;
                }
            }

            scale_apply_exp2(scores, row_max, softmax_scale_log2);
            reduce_sum<false>(scores, row_sum);
        }
    }

    template <typename Tensor0>
    CUTLASS_DEVICE void normalize_softmax(Tensor0 &acc_o) {
        quad_allreduce_(row_sum, row_sum, SumOp{});
        Tensor output = make_tensor(acc_o.data(), convert_layout_rowcol(acc_o.layout()));
        for (int r = 0; r < size<0>(output); ++r) {
            float row_sum_i = 1.f / row_sum(r);
            for (int c = 0; c < size<1>(output); ++c) {
                output(r, c) *= row_sum_i;
            }
        }
    }



};

}
