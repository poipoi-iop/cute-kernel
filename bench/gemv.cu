#include "common.hpp"
#include "cublaslt.hpp"
#include "gemv.cuh"

namespace bench {

// 对每个 shape/精度独立构造输入和 CPU FP64 参考，复用相同的 GPU 数据比较
// 自有 CuTe GEMV、cuBLAS 与 cuBLASLt。所有初始化和搜索都在正式计时之外。
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
    cute_kernel::GemvPlan best_plan;
    CUDA_CHECK(cute_kernel::make_gemv_plan(&best_plan, o.m, o.k, context.prop.multiProcessorCount));
    double best_time = std::numeric_limits<double>::infinity();
    // 搜索线程布局和 K 分区。大 M 已有足够多 CTA 时不再扩大 split，
    // 每个 split 太短时也跳过，避免单纯增加归约开销。
    std::vector<cute_kernel::GemvPlan> candidates;
    if (!o.tune) candidates.push_back(best_plan);
    else for (int group : {4, 8, 16, 32}) for (int warps : {4, 8}) for (int split : {1, 2, 4, 8, 16, 32}) {
        if (split > 1 && (o.k / split < 256 || o.m > context.prop.multiProcessorCount * 16)) continue;
        cute_kernel::GemvPlan p;
        if (cute_kernel::make_gemv_plan(&p, o.m, o.k, context.prop.multiProcessorCount, group, warps, split) == cudaSuccess)
            candidates.push_back(p);
    }
    auto config = [](const cute_kernel::GemvPlan& p) {
        return "group=" + std::to_string(p.group) + ",warps=" + std::to_string(p.warps) + ",splits=" + std::to_string(p.splits);
    };
    // 每个候选都先验证；失败也输出记录，不能因“没被选中”而掩盖错误。
    for (const auto& plan : candidates) {
        Buffer<unsigned char> workspace(plan.workspace_bytes);
        auto fn = [&] { CUDA_CHECK(cute_kernel::launch_gemv(plan, da.ptr, dx.ptr, dy.ptr,
                workspace.ptr, workspace.size, context.stream)); };
        auto accuracy = validate(fn, dy, reference, context.stream, atol, rtol);
        if (!accuracy.passed) {
            emit(o, "gemv_candidate", "correctness_failed", "invalid tuning candidate", config(plan), &accuracy);
            continue;
        }
        double time = measure(fn, context.stream, tuning_options(o)).median;
        if (time < best_time) { best_time = time; best_plan = plan; }
    }
    if (!std::isfinite(best_time)) throw std::runtime_error("no correct GEMV candidate");
    Buffer<unsigned char> workspace(best_plan.workspace_bytes);
    auto fn = [&] { CUDA_CHECK(cute_kernel::launch_gemv(best_plan, da.ptr, dx.ptr, dy.ptr,
            workspace.ptr, workspace.size, context.stream)); };
    evaluate(o, "gemv", fn, dy, reference, context.stream, atol, rtol, config(best_plan), workspace.size);

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
