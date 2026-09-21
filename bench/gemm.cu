#include "common.hpp"
#include "cublaslt.hpp"
#include "legacy.hpp"
#include "hgemm_splitk.cuh"
#include <cutlass/gemm/device/gemm.h>

namespace bench {
using Half = cute::half_t;

static void audit_reference(const Options& o, const std::vector<Half>& a,
    const std::vector<Half>& b, const std::vector<Half>& ref) {
    // Independent FP64 checks of all small outputs and 257 distributed outputs
    // on large GEMMs, including first/last rows and columns.
    std::vector<std::size_t> indices;
    if (ref.size() <= 4096) {
        for (std::size_t i = 0; i < ref.size(); ++i) indices.push_back(i);
    } else {
        indices = {0, std::size_t(o.n - 1), std::size_t(o.m - 1) * o.n, ref.size() - 1};
        for (unsigned i = 0; i < 257; ++i)
            indices.push_back((std::uint64_t(i) * 2654435761ULL + o.seed) % ref.size());
    }
    for (std::size_t index : indices) {
        int m = int(index / o.n), n = int(index % o.n);
        double sum = 0;
        for (int k = 0; k < o.k; ++k)
            sum += double(float(a[std::size_t(m) * o.k + k])) * float(b[std::size_t(n) * o.k + k]);
        if (!std::isfinite(float(ref[index])) || std::abs(float(ref[index]) - sum) > 0.002 + 0.002 * std::abs(sum))
            throw std::runtime_error("cuBLAS reference failed independent FP64 spot check");
    }
}

// 所有 provider 共享输入/输出布局与参考结果。单独报告 splits=1，便于
// 判断收益来自 Split-K 并行化还是基础 GEMM；计时包含最终 reduction。
void run_gemm(Context& context, const Options& o) {
    if (o.dtype != "fp16") throw std::runtime_error("GEMM supports fp16 inputs/output and FP32 accumulation");
    auto a = inputs<Half>(std::size_t(o.m) * o.k, o.seed);
    auto b = inputs<Half>(std::size_t(o.n) * o.k, o.seed + 1);
    Buffer<Half> da(a.size()), db(b.size()), dc(std::size_t(o.m) * o.n);
    da.upload(a); db.upload(b);
    float alpha = 1, beta = 0;
    auto blas = [&] {
        // C(row M,N) == C^T(col N,M) = B(col K,N)^T * A(col K,M).
        BLAS_CHECK(cublasGemmEx(context.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                o.n, o.m, o.k, &alpha, db.ptr, CUDA_R_16F, o.k,
                da.ptr, CUDA_R_16F, o.k, &beta, dc.ptr, CUDA_R_16F, o.n,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    };
    blas(); CUDA_CHECK(cudaStreamSynchronize(context.stream));
    auto reference = dc.download();
    audit_reference(o, a, b, reference);
    evaluate(o, "cublas", blas, dc, reference, context.stream, 0.01, 0.002,
        "GemmEx,FP32_accumulate,default_math,FP32_reduction", Context::kBlasWorkspaceBytes);

    if (context.prop.major >= 8) {
        cute_kernel::HgemmSplitKPlan best_plan;
        CUDA_CHECK(cute_kernel::make_hgemm_splitk_plan(&best_plan, o.m, o.n, o.k, 0, context.prop.multiProcessorCount));
        double best = std::numeric_limits<double>::infinity();
        std::vector<int> candidates = o.tune ? std::vector<int>{1, 2, 4, 8, 16, 32} : std::vector<int>{best_plan.splits};
        // 给自有 Split-K 搜索设置与工业基线相同的 32 MiB workspace 上限。
        // 创建候选/分配/正确性验证/短跑均不进入最终 evaluate() 的采样。
        for (int splits : candidates) {
            cute_kernel::HgemmSplitKPlan plan;
            if (cute_kernel::make_hgemm_splitk_plan(&plan, o.m, o.n, o.k, splits) != cudaSuccess) continue;
            if (plan.workspace_bytes > Context::kBlasWorkspaceBytes) continue;
            Buffer<unsigned char> workspace(plan.workspace_bytes);
            auto fn = [&] { CUDA_CHECK(cute_kernel::launch_hgemm_splitk(plan, dc.ptr, da.ptr, db.ptr,
                    workspace.ptr, workspace.size, context.stream)); };
            auto accuracy = validate(fn, dc, reference, context.stream, 0.01, 0.002);
            if (!accuracy.passed) {
                emit(o, "hgemm_splitk_candidate", "correctness_failed", "invalid tuning candidate", "splits=" + std::to_string(splits), &accuracy);
                continue;
            }
            double time = measure(fn, context.stream, tuning_options(o)).median;
            if (time < best) { best = time; best_plan = plan; }
        }
        if (!std::isfinite(best)) throw std::runtime_error("no correct Split-K candidate");
        Buffer<unsigned char> workspace(best_plan.workspace_bytes);
        auto fn = [&] { CUDA_CHECK(cute_kernel::launch_hgemm_splitk(best_plan, dc.ptr, da.ptr, db.ptr,
                workspace.ptr, workspace.size, context.stream)); };
        evaluate(o, "hgemm_splitk", fn, dc, reference, context.stream, 0.01, 0.002,
            "tile=64x64x32,stages=3,splits=" + std::to_string(best_plan.splits), workspace.size);
        cute_kernel::HgemmSplitKPlan single;
        CUDA_CHECK(cute_kernel::make_hgemm_splitk_plan(&single, o.m, o.n, o.k, 1));
        auto single_fn = [&] { CUDA_CHECK(cute_kernel::launch_hgemm_splitk(single, dc.ptr, da.ptr, db.ptr, nullptr, 0, context.stream)); };
        evaluate(o, "hgemm_splitk_s1", single_fn, dc, reference, context.stream, 0.01, 0.002, "splits=1");
    } else {
        emit(o, "hgemm_splitk", "unsupported", "requires SM80 or newer");
        emit(o, "hgemm_splitk_s1", "unsupported", "requires SM80 or newer");
    }

    LtMatmul lt(o.m, o.n, o.k, CUDA_R_16F);
    if (lt.select(da.ptr, db.ptr, dc, reference, context.stream, o, 0.01, 0.002)) {
        auto fn = [&] { lt.launch(da.ptr, db.ptr, dc.ptr, context.stream); };
        evaluate(o, "cublaslt", fn, dc, reference, context.stream, 0.01, 0.002, lt.config(), lt.workspace_bytes());
    } else emit(o, "cublaslt", "unsupported", "no valid heuristic algorithm");

    // 这里仅代表一个明确记录的 CUTLASS SM80 配置，不代表其全部最优配置。
    // B 使用 ColumnMajor(K,N)，正好对应原始连续内存 B[N,K]，无需转置。
    using Cutlass = cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::RowMajor, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
    Cutlass cutlass_op;
    typename Cutlass::Arguments args({o.m, o.n, o.k}, {da.ptr, o.k}, {db.ptr, o.k},
        {dc.ptr, o.n}, {dc.ptr, o.n}, {1, 0});
    auto cutlass_status = cutlass_op.can_implement(args);
    if (context.prop.major >= 8 && cutlass_status == cutlass::Status::kSuccess) {
        Buffer<unsigned char> workspace(cutlass_op.get_workspace_size(args));
        auto check = [](cutlass::Status status) {
            if (status != cutlass::Status::kSuccess)
                throw std::runtime_error(std::string("CUTLASS: ") + cutlassGetStatusString(status));
        };
        check(cutlass_op.initialize(args, workspace.ptr, context.stream));
        auto fn = [&] { check(cutlass_op(context.stream)); };
        evaluate(o, "cutlass_sm80", fn, dc, reference, context.stream, 0.01, 0.002,
            "tile=128x128x32,stages=3,FP32_accumulate,fixed_configuration", workspace.size);
    } else emit(o, "cutlass_sm80", "unsupported", "SM80/alignment requirements of the selected CUTLASS configuration");

    Legacy original(o.legacy_dir + "/liboriginal_hgemm.so");
    if (!original.available()) emit(o, "original_hgemm", "unavailable", "see build_manifest.json and original_hgemm.log");
    else {
        auto status = original.launch(da.ptr, db.ptr, dc.ptr, o.m, o.n, o.k, context.stream);
        if (status == cudaErrorNotSupported) emit(o, "original_hgemm", "unsupported", "original tile/pipeline shape constraints");
        else {
            CUDA_CHECK(status);
            auto fn = [&] { CUDA_CHECK(original.launch(da.ptr, db.ptr, dc.ptr, o.m, o.n, o.k, context.stream)); };
            evaluate(o, "original_hgemm", fn, dc, reference, context.stream, 0.01, 0.002, "unchanged_original,FP16_accumulate");
        }
    }
}
}  // namespace bench
