#include "common.hpp"
#include "legacy.hpp"

namespace bench {
// 通过 ABI 适配器调用原始 normalization kernel；输入与 Python 工业基线
// 使用同一生成公式。CPU FP64 参考在计时之外计算，错误结果只记录诊断。
void run_legacy(Context& context, const Options& o) {
    if (o.dtype != "fp32") throw std::runtime_error("original normalization kernels use fp32");
    std::string provider = "original_" + o.op;
    Legacy original(o.legacy_dir + "/lib" + provider + ".so");
    if (!original.available()) {
        emit(o, provider, "unavailable", "see build_manifest.json and " + provider + ".log");
        return;
    }
    auto input = inputs<float>(std::size_t(o.m) * o.n, o.seed, 1.0f);
    Buffer<float> x(input.size()), y(input.size());
    x.upload(input);
    auto status = original.launch(x.ptr, nullptr, y.ptr, o.m, o.n, 0, context.stream);
    if (status == cudaErrorNotSupported) {
        emit(o, provider, "unsupported", "uncompiled COL or unaligned original float4 row stride"); return;
    }
    CUDA_CHECK(status);
    std::vector<double> ref(input.size());
    for (int r = 0; r < o.m; ++r) {
        auto offset = std::size_t(r) * o.n;
        if (o.op == "rms_norm") {
            double ss = 0;
            for (int j = 0; j < o.n; ++j) ss += double(input[offset + j]) * input[offset + j];
            double scale = 1.0 / std::sqrt(ss / o.n + 1e-5);
            for (int j = 0; j < o.n; ++j) ref[offset + j] = input[offset + j] * scale;
        } else {
            double max = -std::numeric_limits<double>::infinity(), sum = 0;
            for (int j = 0; j < o.n; ++j) max = std::max(max, double(input[offset + j]));
            for (int j = 0; j < o.n; ++j) sum += std::exp(double(input[offset + j]) - max);
            for (int j = 0; j < o.n; ++j) ref[offset + j] = std::exp(double(input[offset + j]) - max) / sum;
        }
    }
    auto fn = [&] { CUDA_CHECK(original.launch(x.ptr, nullptr, y.ptr, o.m, o.n, 0, context.stream)); };
    evaluate(o, provider, fn, y, ref, context.stream, 1e-6, 1e-4, "unchanged_original,threads=256");
}
}  // namespace bench
