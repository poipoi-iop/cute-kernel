#include "common.hpp"

// 原生 runner 每次执行一个 shape，并逐行输出 JSON。Python 层负责 suite、
// 子进程隔离与持久化；退出码 77 明确表示没有 GPU，不能当作测试通过。
int main(int argc, char** argv) {
    bench::Options o;
    try {
        bool describe = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--device-info") { describe = true; continue; }
            if (arg == "--no-tune") { o.tune = false; continue; }
            if (arg == "--help") {
                std::cout << "cute_bench --op gemm|gemv|softmax|online_softmax|rms_norm --dtype fp16|fp32\n"
                    "  --m M --n N --k K --device 0 --timing graph|events --warmup 10\n"
                    "  --iterations 50 --samples 9 --legacy-dir BUILD --no-tune\n";
                return 0;
            }
            if (i + 1 >= argc) throw std::runtime_error("missing argument for " + arg);
            std::string value = argv[++i];
            if (arg == "--op") o.op = value;
            else if (arg == "--dtype") o.dtype = value;
            else if (arg == "--timing") o.timing = value;
            else if (arg == "--legacy-dir") o.legacy_dir = value;
            else if (arg == "--m") o.m = std::stoi(value);
            else if (arg == "--n") o.n = std::stoi(value);
            else if (arg == "--k") o.k = std::stoi(value);
            else if (arg == "--device") o.device = std::stoi(value);
            else if (arg == "--warmup") o.warmup = std::stoi(value);
            else if (arg == "--iterations") o.iterations = std::stoi(value);
            else if (arg == "--samples") o.samples = std::stoi(value);
            else if (arg == "--seed") o.seed = std::stoi(value);
            else throw std::runtime_error("unknown option: " + arg);
        }
        if (o.m <= 0 || o.n <= 0 || o.k < 0 || o.warmup < 0 || o.iterations < 1 || o.samples < 1 ||
            (o.timing != "graph" && o.timing != "events")) throw std::runtime_error("invalid benchmark arguments");
        if ((o.op == "gemm" || o.op == "gemv") && o.k == 0)
            throw std::runtime_error("zero-K is a correctness test, not a throughput benchmark");
        int count = 0;
        auto status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess || count == 0) {
            bench::emit(o, "environment", "no_gpu", cudaGetErrorString(status)); return 77;
        }
        bench::Context context(o.device);
        if (describe) {
            int runtime = 0, driver = 0, blas_version = 0;
            CUDA_CHECK(cudaRuntimeGetVersion(&runtime)); CUDA_CHECK(cudaDriverGetVersion(&driver));
            BLAS_CHECK(cublasGetVersion(context.blas, &blas_version));
            std::cout << "{\"status\":\"ok\",\"device\":" << o.device
                << ",\"gpu\":" << bench::quote(context.prop.name)
                << ",\"sm\":" << context.prop.major * 10 + context.prop.minor
                << ",\"multiprocessors\":" << context.prop.multiProcessorCount
                << ",\"total_memory_bytes\":" << context.prop.totalGlobalMem
                << ",\"l2_bytes\":" << context.prop.l2CacheSize
                << ",\"cuda_runtime\":" << runtime << ",\"cuda_driver\":" << driver
                << ",\"cublas_version\":" << blas_version << "}\n";
        } else if (o.op == "gemm") bench::run_gemm(context, o);
        else if (o.op == "gemv") bench::run_gemv(context, o);
        else if (o.op == "softmax" || o.op == "online_softmax" || o.op == "rms_norm") bench::run_legacy(context, o);
        else throw std::runtime_error("unknown operator: " + o.op);
        return 0;
    } catch (const std::exception& e) {
        bench::emit(o, "runner", "runtime_error", e.what());
        return 1;
    }
}
