#include "common.hpp"
#include "cublaslt.hpp"
#include "gemv.cuh"

namespace bench {

// 对每个 shape/精度独立构造输入和 CPU FP64 参考，复用相同的 GPU 数据比较
// 自有 GEMV、cuBLAS 与 cuBLASLt。所有初始化都在正式计时之外。
template <class T>
void gemv_case(Context& context, const Options& o) {
    auto a = inputs<T>(std::size_t(o.m) * o.k, o.seed);
    auto x = inputs<T>(o.k, o.seed + 1);
    std::vector<double> reference(o.m, 0);
    for (int row = 0; row < o.m; ++row)
        for (int k = 0; k < o.k; ++k)
            reference[row] += double(as_float(a[std::size_t(row) * o.k + k])) * as_float(x[k]);
    Buffer<T> da(a.size()), dx(x.size()), dy(o.m);
    da.upload(a); dx.upload(x);
    constexpr bool half = std::is_same_v<T, __half>;
    double atol = half ? 0.002 : 2e-5, rtol = half ? 0.002 : 2e-4;
    // 自有 GEMV 不做运行时调参：gemv() 内部按 M/K 选择线程映射与 Split-K。
    // workspace 只需在分配后清零一次，kernel 会自行把计数器归零。
    Buffer<unsigned char> workspace(cute_kernel::gemv_workspace_bytes(o.m));
    CUDA_CHECK(cudaMemset(workspace.ptr, 0, workspace.size));
    auto fn = [&] { CUDA_CHECK(cute_kernel::gemv(da.ptr, dx.ptr, dy.ptr, o.m, o.k, context.stream, workspace.ptr)); };
    evaluate(o, "gemv", fn, dy, reference, context.stream, atol, rtol, "heuristic", workspace.size);

    // cuBLAS 按列主序解释 A：原行主序 [M,K] 等价于列主序 [K,M]。
    // Sgemv 取 transpose，FP16 使用同一数学语义的 GemmEx(M,1,K)。
    float alpha = 1, beta = 0;
    auto blas = [&] {
        if constexpr (!half)
            BLAS_CHECK(cublasSgemv(context.blas, CUBLAS_OP_T, o.k, o.m, &alpha,
                    da.ptr, o.k, dx.ptr, 1, &beta, dy.ptr, 1));
        else
            BLAS_CHECK(cublasGemmEx(context.blas, CUBLAS_OP_T, CUBLAS_OP_N, o.m, 1, o.k,
                    &alpha, da.ptr, CUDA_R_16F, o.k, dx.ptr, CUDA_R_16F, o.k,
                    &beta, dy.ptr, CUDA_R_16F, o.m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    };
    evaluate(o, "cublas", blas, dy, reference, context.stream, atol, rtol,
        half ? "GemmEx,N=1,FP32_accumulate,FP32_reduction" : "Sgemv,default_math", Context::kBlasWorkspaceBytes);
    LtMatmul lt(o.m, 1, o.k, half ? CUDA_R_16F : CUDA_R_32F);
    if (lt.select(da.ptr, dx.ptr, dy, reference, context.stream, o, atol, rtol)) {
        auto lt_fn = [&] { lt.launch(da.ptr, dx.ptr, dy.ptr, context.stream); };
        evaluate(o, "cublaslt", lt_fn, dy, reference, context.stream, atol, rtol, lt.config(), lt.workspace_bytes());
    } else emit(o, "cublaslt", "unsupported", "no valid heuristic algorithm");
}

void run_gemv(Context& context, const Options& o) {
    if (o.dtype == "fp16") gemv_case<__half>(context, o);
    else if (o.dtype == "fp32") gemv_case<float>(context, o);
    else throw std::runtime_error("GEMV dtype must be fp16 or fp32");
}
}  // namespace bench
