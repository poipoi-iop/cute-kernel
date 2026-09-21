#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench {

inline void cuda_check(cudaError_t e, const char* call) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(e));
}
inline void blas_check(cublasStatus_t e, const char* call) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(call) + ": cuBLAS status " + std::to_string(int(e)));
}
#define CUDA_CHECK(call) ::bench::cuda_check((call), #call)
#define BLAS_CHECK(call) ::bench::blas_check((call), #call)

// 仅用于 benchmark setup/validation 的 device buffer 所有者。
// 拷贝构造被禁用，防止重复 cudaFree；upload/download 使用同步拷贝，
// 调用位置均在 measure() 之外，不计入 kernel 延迟。
template <class T>
class Buffer {
public:
    T* ptr = nullptr;
    std::size_t size = 0;
    explicit Buffer(std::size_t count = 0) : size(count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::runtime_error("buffer size overflow");
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    }
    ~Buffer() { if (ptr) cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& x) {
        if (x.size() != size) throw std::runtime_error("upload size mismatch");
        if (size) CUDA_CHECK(cudaMemcpy(ptr, x.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> x(size);
        if (size) CUDA_CHECK(cudaMemcpy(x.data(), ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
        return x;
    }
};

// 每个评测进程持有独立的非默认 stream、cuBLAS handle 和显式 workspace。
// 自有算子与工业基线都提交到这个 stream，避免默认 stream 的隐式同步
// 改变比较条件；handle/分配的初始化成本同样不进入 measure()。
struct Context {
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    void* blas_workspace = nullptr;
    static constexpr std::size_t kBlasWorkspaceBytes = 32 * 1024 * 1024;
    cudaDeviceProp prop{};
    explicit Context(int device) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        BLAS_CHECK(cublasCreate(&blas));
        BLAS_CHECK(cublasSetStream(blas, stream));
        // Keep Tensor Core selection enabled, but require FP32 intermediate
        // reductions for FP16 output. FAST_TF32 is never requested for FP32 input.
        BLAS_CHECK(cublasSetMathMode(blas, static_cast<cublasMath_t>(
                    CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
        CUDA_CHECK(cudaMalloc(&blas_workspace, kBlasWorkspaceBytes));
        BLAS_CHECK(cublasSetWorkspace(blas, blas_workspace, kBlasWorkspaceBytes));
    }
    ~Context() {
        if (blas) cublasDestroy(blas);
        if (blas_workspace) cudaFree(blas_workspace);
        if (stream) cudaStreamDestroy(stream);
    }
};

// 一次 shape 的计时设置；完整配置会由 Python runner 保存在 metadata 中。
struct Options {
    std::string op = "gemm", dtype = "fp16", timing = "graph", legacy_dir;
    int m = 128, n = 128, k = 4096, device = 0;
    int warmup = 10, iterations = 50, samples = 9, seed = 17;
    bool tune = true;
};

// 只转义 JSON 字符串，不参与 shell 命令构造。诊断消息可能包含换行与引号，
// 必须转义后写入 JSON Lines，否则 Python runner 无法可靠解析失败记录。
inline std::string quote(const std::string& s) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    return out.str() + '"';
}

struct Stats { double median = 0, minimum = 0, p95 = 0; std::vector<double> samples; };

// Capture the complete operation, including all Split-K reduction launches.
// Allocation, transfers, correctness, algorithm search and warmup are excluded.
// Graph 中捕获 iterations 次完整 fn，每次 replay 的 event 时间再除以
// iterations，得到单次操作耗时。events 模式直接提交相同数量的 fn；
// 两种模式不混用，以免把 host 提交间隙与设备执行时间混成一组样本。
inline Stats measure(const std::function<void()>& fn, cudaStream_t stream, const Options& o) {
    for (int i = 0; i < o.warmup; ++i) fn();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    try {
        if (o.timing == "graph") {
            CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            try { for (int i = 0; i < o.iterations; ++i) fn(); }
            catch (...) { cudaStreamEndCapture(stream, &graph); if (graph) cudaGraphDestroy(graph); graph = nullptr; throw; }
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        Stats stats;
        for (int s = 0; s < o.samples; ++s) {
            CUDA_CHECK(cudaEventRecord(start, stream));
            if (executable) CUDA_CHECK(cudaGraphLaunch(executable, stream));
            else for (int i = 0; i < o.iterations; ++i) fn();
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            stats.samples.push_back(ms * 1000.0 / o.iterations);
        }
        auto sorted = stats.samples;
        std::sort(sorted.begin(), sorted.end());
        stats.minimum = sorted.front();
        stats.median = (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) / 2;
        stats.p95 = sorted[std::size_t(std::ceil(0.95 * sorted.size())) - 1];
        if (!(stats.minimum > 0)) throw std::runtime_error("timer resolution insufficient; increase iterations");
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        return stats;
    } catch (...) {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        throw;
    }
}

// 调参使用较短的独立采样；胜出配置在 evaluate() 中重新计时，避免直接把
// 搜索过程中偶然最快的样本当成正式性能结果。
inline Options tuning_options(Options o) {
    o.warmup = 3; o.iterations = 12; o.samples = 3; return o;
}

// Reproducible input, also implemented in torch_baselines.py.
inline float input_value(std::size_t i, unsigned seed) {
    std::uint32_t h = (std::uint32_t(i) ^ seed) * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28) + 4)) ^ h) * 277803737u;
    h = (h >> 22) ^ h;
    return float(int(h & 65535) - 32768) / 32768.0f;
}
template <class T>
inline float as_float(T v) { return float(v); }
template <>
inline float as_float(__half v) { return __half2float(v); }
template <class T>
inline std::vector<T> inputs(std::size_t count, int seed, float scale = 0.25f) {
    std::vector<T> out(count);
    for (std::size_t i = 0; i < count; ++i) out[i] = T(input_value(i, seed) * scale);
    return out;
}

struct Accuracy {
    bool passed = true;
    double max_abs = 0, max_scaled = 0, rmse = 0;
    std::size_t checked = 0;
};

// 全输出比较采用 atol + rtol*abs(ref)，同时记录最大绝对误差、相对阈值
// 倍数和 RMSE。NaN/Inf 一律失败，不能借由 NaN 比较结果绕过检查。
template <class T, class U>
Accuracy compare(const std::vector<T>& result, const std::vector<U>& reference,
    double atol, double rtol) {
    if (result.size() != reference.size()) throw std::runtime_error("reference size mismatch");
    Accuracy a;
    a.checked = result.size();
    double squared = 0;
    for (std::size_t i = 0; i < result.size(); ++i) {
        double got = as_float(result[i]), ref = double(reference[i]);
        if (!std::isfinite(got) || !std::isfinite(ref)) {
            a.passed = false;
            a.max_abs = a.max_scaled = a.rmse = std::numeric_limits<double>::infinity();
            return a;
        }
        double err = std::abs(got - ref), scaled = err / (atol + rtol * std::abs(ref));
        a.max_abs = std::max(a.max_abs, err); a.max_scaled = std::max(a.max_scaled, scaled);
        squared += err * err;
        if (scaled > 1) a.passed = false;
    }
    a.rmse = std::sqrt(squared / std::max<std::size_t>(1, result.size()));
    return a;
}

// 标准 JSON 不支持 NaN/Infinity；无穷误差用 null 表示，失败状态另行记录。
inline void number(std::ostream& out, double v) {
    if (std::isfinite(v)) out << std::setprecision(10) << v; else out << "null";
}

// 一条记录只描述一个 provider/shape。失败记录没有 stats，因此不会输出
// 伪造的 0 us。effective_gbps 使用逻辑字节数，不能解读为实测 HBM 带宽。
inline void emit(const Options& o, const std::string& provider, const std::string& status,
    const std::string& note = "", const std::string& config = "",
    const Accuracy* accuracy = nullptr, const Stats* stats = nullptr,
    std::size_t workspace = 0) {
    std::cout << "{\"op\":" << quote(o.op) << ",\"dtype\":" << quote(o.dtype)
        << ",\"m\":" << o.m << ",\"n\":" << o.n << ",\"k\":" << o.k
        << ",\"provider\":" << quote(provider) << ",\"status\":" << quote(status)
        << ",\"note\":" << quote(note) << ",\"config\":" << quote(config)
        << ",\"timing\":" << quote(o.timing) << ",\"cache\":\"warm_reuse\""
        << ",\"workspace_bytes\":" << workspace;
    if (accuracy) {
        std::cout << ",\"checked_elements\":" << accuracy->checked << ",\"max_abs_error\":";
        number(std::cout, accuracy->max_abs);
        std::cout << ",\"max_scaled_error\":"; number(std::cout, accuracy->max_scaled);
        std::cout << ",\"rmse\":"; number(std::cout, accuracy->rmse);
    }
    if (stats) {
        double flops = o.op == "gemm" ? 2.0 * o.m * o.n * o.k : 2.0 * o.m * o.k;
        double item_size = o.dtype == "fp16" ? 2 : 4;
        double bytes = o.op == "gemm" ? item_size * ((double(o.m) + o.n) * o.k + double(o.m) * o.n)
            : o.op == "gemv" ? item_size * (double(o.m) * o.k + o.k + o.m)
            : 2 * item_size * double(o.m) * o.n;
        std::cout << ",\"median_us\":"; number(std::cout, stats->median);
        std::cout << ",\"min_us\":"; number(std::cout, stats->minimum);
        std::cout << ",\"p95_us\":"; number(std::cout, stats->p95);
        std::cout << ",\"tflops\":";
        if (o.op == "gemm" || o.op == "gemv") number(std::cout, flops / (stats->median * 1e6));
        else std::cout << "null";
        std::cout << ",\"effective_gbps\":"; number(std::cout, bytes / (stats->median * 1000));
        std::cout << ",\"samples_us\":[";
        for (std::size_t i = 0; i < stats->samples.size(); ++i) {
            if (i) std::cout << ',';
            number(std::cout, stats->samples[i]);
        }
        std::cout << ']';
    }
    std::cout << "}\n" << std::flush;
}

// 用全 1 bit pattern 将输出毒化为 NaN：漏写输出也会被 compare 检出。
// 同步和 D2H 下载属于正确性验证，不属于 measure() 的时间区间。
template <class T, class U>
Accuracy validate(const std::function<void()>& fn, Buffer<T>& out,
    const std::vector<U>& reference, cudaStream_t stream,
    double atol, double rtol) {
    if (out.size) CUDA_CHECK(cudaMemsetAsync(out.ptr, 0xff, out.size * sizeof(T), stream));
    fn();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return compare(out.download(), reference, atol, rtol);
}

// 所有 provider 通过同一条“验证 -> 计时 -> 记录”路径。数值失败立即
// 保留诊断记录并停止该 provider 的计时，避免错误实现进入性能图。
template <class T, class U>
bool evaluate(const Options& o, const std::string& provider, const std::function<void()>& fn,
    Buffer<T>& out, const std::vector<U>& ref, cudaStream_t stream,
    double atol, double rtol, const std::string& config = "",
    std::size_t workspace = 0) {
    auto accuracy = validate(fn, out, ref, stream, atol, rtol);
    if (!accuracy.passed) {
        emit(o, provider, "correctness_failed", "timing withheld", config, &accuracy);
        return false;
    }
    auto stats = measure(fn, stream, o);
    emit(o, provider, "ok", "", config, &accuracy, &stats, workspace);
    return true;
}

void run_gemm(Context&, const Options&);
void run_gemv(Context&, const Options&);
void run_legacy(Context&, const Options&);

}  // namespace bench
