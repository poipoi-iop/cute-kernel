#pragma once
#include "common.hpp"
#include <cublasLt.h>

namespace bench {

// Row-major A[M,K], B[N,K] with op(B)=T; GEMV is exactly N=1.
// descriptor、heuristic 搜索与 workspace 分配只在 setup 进行。
// select() 先过滤不支持/不正确的候选，再短跑选优；launch() 复用选中的
// algo，可被 measure() 反复调用或捕获进 CUDA Graph。
class LtMatmul {
    cublasLtHandle_t handle_ = nullptr;
    cublasLtMatmulDesc_t desc_ = nullptr;
    cublasLtMatrixLayout_t a_ = nullptr, b_ = nullptr, c_ = nullptr;
    cublasLtMatmulPreference_t preference_ = nullptr;
    std::vector<cublasLtMatmulHeuristicResult_t> algorithms_;
    Buffer<unsigned char> workspace_;
    int selected_ = -1;
public:
    explicit LtMatmul(int m, int n, int k, cudaDataType_t dtype)
        : workspace_(32 * 1024 * 1024) {
            BLAS_CHECK(cublasLtCreate(&handle_));
            BLAS_CHECK(cublasLtMatmulDescCreate(&desc_, CUBLAS_COMPUTE_32F, CUDA_R_32F));
            // B 在内存中是 [N,K] 行主序，而数学乘法需要 [K,N]，因此只设置
            // op(B)=T，不做额外转置拷贝。N=1 时同一接口恰好表示 GEMV。
            cublasOperation_t transpose = CUBLAS_OP_T;
            BLAS_CHECK(cublasLtMatmulDescSetAttribute(desc_, CUBLASLT_MATMUL_DESC_TRANSB, &transpose, sizeof(transpose)));
            BLAS_CHECK(cublasLtMatrixLayoutCreate(&a_, dtype, m, k, k));
            BLAS_CHECK(cublasLtMatrixLayoutCreate(&b_, dtype, n, k, k));
            BLAS_CHECK(cublasLtMatrixLayoutCreate(&c_, dtype, m, n, n));
            cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
            for (auto layout : {a_, b_, c_})
                BLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
            BLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference_));
            std::size_t bytes = workspace_.size;
            BLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes)));
            algorithms_.resize(16);
            int count = 0;
            auto status = cublasLtMatmulAlgoGetHeuristic(handle_, desc_, a_, b_, c_, c_, preference_,
                int(algorithms_.size()), algorithms_.data(), &count);
            if (status != CUBLAS_STATUS_NOT_SUPPORTED && status != CUBLAS_STATUS_ARCH_MISMATCH)
                BLAS_CHECK(status);
            algorithms_.resize(count);
        }
    LtMatmul(const LtMatmul&) = delete;
    LtMatmul& operator=(const LtMatmul&) = delete;
    ~LtMatmul() {
        if (preference_) cublasLtMatmulPreferenceDestroy(preference_);
        if (a_) cublasLtMatrixLayoutDestroy(a_);
        if (b_) cublasLtMatrixLayoutDestroy(b_);
        if (c_) cublasLtMatrixLayoutDestroy(c_);
        if (desc_) cublasLtMatmulDescDestroy(desc_);
        if (handle_) cublasLtDestroy(handle_);
    }
    cublasStatus_t try_launch(const void* a, const void* b, void* c, cudaStream_t stream) {
        if (selected_ < 0) return CUBLAS_STATUS_NOT_SUPPORTED;
        float alpha = 1, beta = 0;
        return cublasLtMatmul(handle_, desc_, &alpha, a, a_, b, b_, &beta, c, c_, c, c_,
            &algorithms_[selected_].algo, workspace_.ptr, workspace_.size, stream);
    }
    void launch(const void* a, const void* b, void* c, cudaStream_t stream) {
        BLAS_CHECK(try_launch(a, b, c, stream));
    }
    // 不能只接受 heuristic 的第一个条目：实际指针/stride 仍可能不支持，
    // 且 heuristic 排序不等于当前 shape 上的实测排序。
    template <class T, class U>
    bool select(const T* a, const T* b, Buffer<T>& c, const std::vector<U>& ref,
        cudaStream_t stream, const Options& options, double atol, double rtol) {
        double best = std::numeric_limits<double>::infinity();
        int best_index = -1;
        for (int i = 0; i < int(algorithms_.size()); ++i) {
            if (algorithms_[i].state != CUBLAS_STATUS_SUCCESS) continue;
            selected_ = i;
            auto fn = [&] { launch(a, b, c.ptr, stream); };
            CUDA_CHECK(cudaMemsetAsync(c.ptr, 0xff, c.size * sizeof(T), stream));
            auto status = try_launch(a, b, c.ptr, stream);
            if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH) continue;
            BLAS_CHECK(status);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (!compare(c.download(), ref, atol, rtol).passed) continue;
            double time = measure(fn, stream, tuning_options(options)).median;
            if (time < best) { best = time; best_index = i; }
            if (!options.tune) break;
        }
        selected_ = best_index;
        return selected_ >= 0;
    }
    // 报告分配的容量；算法实际要求的字节数另外放在 config() 中。
    std::size_t workspace_bytes() const {
        return workspace_.size;
    }
    std::string config() const {
        int id = -1, splits = 0;
        std::size_t written = 0;
        if (selected_ >= 0) {
            cublasLtMatmulAlgoConfigGetAttribute(&algorithms_[selected_].algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &written);
            cublasLtMatmulAlgoConfigGetAttribute(&algorithms_[selected_].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splits, sizeof(splits), &written);
        }
        auto required = selected_ >= 0 ? algorithms_[selected_].workspaceSize : 0;
        return "algorithm=" + std::to_string(id) + ",split_k=" + std::to_string(splits) +
            ",heuristic_candidates=" + std::to_string(algorithms_.size()) + ",required_workspace=" + std::to_string(required);
    }
};

}  // namespace bench
