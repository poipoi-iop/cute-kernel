#include "common.hpp"
#include "hgemm_splitk.cuh"
#include "gemv.cuh"
#include <array>
#include <cstring>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// 这些契约测试不执行 CUDA kernel，因而无 GPU 时也能验证 tile 约束、workspace
// 尺寸与空 split 路径；真正的数值和越界检查在后面的 device 测试中完成。
void host_tests() {
    using namespace hgemm_splitk;
    float tiny[1];
    require(hgemm_splitk_launch(65, 128, 32, nullptr, nullptr, nullptr) == cudaErrorNotSupported,
        "M tail rejected");
    require(hgemm_splitk_launch(128, 129, 32, nullptr, nullptr, nullptr) == cudaErrorNotSupported,
        "N tail rejected");
    require(hgemm_splitk_launch(128, 128, 33, nullptr, nullptr, nullptr) == cudaErrorNotSupported,
        "K tail rejected");
    require(!can_launch(0, 64, 64) && !can_launch(64, 64, 0), "empty shape rejected");
    require(can_launch(64, 64, 32) && can_launch(1024, 512, 4096), "whole tiles accepted");
    require(hgemm_splitk_workspace_bytes(65, 129, 3) == std::size_t(65) * 129 * 3 * 4,
        "workspace size");
    require(hgemm_splitk_workspace_bytes(64, 64, 1) == 0, "no workspace for one split");
    require(gemv<float>(tiny, tiny, tiny, -1, 8) == cudaErrorInvalidValue, "negative gemv M");
    CUDA_CHECK(gemv<float>(nullptr, nullptr, nullptr, 0, 33));
    std::cout << "PASS: host API validation and workspace sizing\n";
}

// 随机输入外再检查全零与大幅度抵消，避免只在温和数据上验证浮点归约。
template <class T>
void pattern(std::vector<T>& x, int mode) {
    if (mode == 1) for (auto& v : x) v = T(0);
    if (mode == 2) for (std::size_t i = 0; i < x.size(); ++i) x[i] = T(i % 2 ? -64.0f : 64.0f);
}

// 输出前后各保留一个 sentinel，workspace 前后各保留四个 float；执行后
// 同时核对 FP64 参考、guard 和重复执行的逐 bit 一致性。K=32 的固定形状
// 另外检查 K 区间短于流水线时每个 split 仍写满自己的 partial slab。
void test_gemm(cudaStream_t stream, int m, int n, int k, int splits, int offset, int mode) {
    using T = cute::half_t;
    std::size_t workspace_bytes = hgemm_splitk::hgemm_splitk_workspace_bytes(m, n, splits);
    auto a = bench::inputs<T>(std::size_t(m) * k + offset, 31);
    auto b = bench::inputs<T>(std::size_t(n) * k + offset, 73);
    pattern(a, mode);
    if (mode == 2) for (auto& v : b) v = T(1);
    bench::Buffer<T> da(a.size()), db(b.size()), dc(std::size_t(m) * n + 2);
    da.upload(a); db.upload(b);
    std::vector<T> initialized(dc.size, T(-13)); dc.upload(initialized);
    bench::Buffer<float> workspace(workspace_bytes / 4 + 8);
    workspace.upload(std::vector<float>(workspace.size, -1234));
    auto fn = [&] { CUDA_CHECK(hgemm_splitk::hgemm_splitk_launch(m, n, k,
            da.ptr ? da.ptr + offset : nullptr, db.ptr ? db.ptr + offset : nullptr,
            dc.ptr + 1, workspace_bytes ? workspace.ptr + 4 : nullptr, splits, stream)); };
    fn(); CUDA_CHECK(cudaStreamSynchronize(stream));
    auto result = dc.download();
    require(float(result.front()) == -13 && float(result.back()) == -13, "GEMM output guard modified");
    std::vector<double> ref(std::size_t(m) * n, 0);
    for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col)
        for (int q = 0; q < k; ++q)
            ref[std::size_t(row) * n + col] += double(float(a[offset + std::size_t(row) * k + q])) * float(b[offset + std::size_t(col) * k + q]);
    std::vector<T> actual(result.begin() + 1, result.end() - 1);
    require(bench::compare(actual, ref, 0.002, 0.002).passed,
        "GEMM mismatch: " + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k) + ", splits=" + std::to_string(splits));
    fn(); CUDA_CHECK(cudaStreamSynchronize(stream));
    auto repeat = dc.download();
    require(std::memcmp(result.data(), repeat.data(), result.size() * sizeof(T)) == 0, "GEMM nondeterministic");
    auto guard = workspace.download();
    for (std::size_t i = 0; i < guard.size(); ++i)
        if (i < 4 || i >= guard.size() - 4) require(guard[i] == -1234, "GEMM workspace guard modified");
    if (m == 65 && n == 129 && k == 97 && splits == 3 && !offset && !mode) {
        bench::Options options; options.iterations = 3; options.samples = 2; options.warmup = 1;
        bench::measure(fn, stream, options);
        auto captured = dc.download();
        require(std::memcmp(result.data(), captured.data(), result.size() * sizeof(T)) == 0, "GEMM graph replay mismatch");
    }
}

// 逐个运行 gemv() 可能分派到的全部 (G, THREADS) 实例，以及 gemv() 本身，
// 用行尾与 K 尾部 shape 验证越界处理；一个固定案例还检查 Graph 重放。
template <class T>
void test_gemv(cudaStream_t stream, int m, int k, int offset, int mode) {
    auto a = bench::inputs<T>(std::size_t(m) * k + offset, 19);
    auto x = bench::inputs<T>(k + offset, 89);
    pattern(a, mode);
    if (mode == 2) for (auto& v : x) v = T(1);
    bench::Buffer<T> da(a.size()), dx(x.size()), dy(m + 2);
    da.upload(a); dx.upload(x);
    std::vector<double> ref(m, 0);
    for (int r = 0; r < m; ++r) for (int q = 0; q < k; ++q)
        ref[r] += double(bench::as_float(a[offset + std::size_t(r) * k + q])) * bench::as_float(x[offset + q]);
    const T* pa = da.ptr ? da.ptr + offset : nullptr;
    const T* px = dx.ptr ? dx.ptr + offset : nullptr;
    T* py = dy.ptr + 1;
    bool vec = k % (16 / sizeof(T)) == 0 && !offset;
    using Launch = std::function<void()>;
    // workspace 前后各留 4 个 guard 字；counter 部分初始为 0。
    std::size_t words = cute_kernel::gemv_workspace_bytes(m) / 4;
    bench::Buffer<unsigned> workspace(words + 8);
    std::vector<unsigned> init(workspace.size, 0xdeadbeefu);
    std::fill(init.begin() + 4, init.end() - 4, 0u);
    workspace.upload(init);
    float* partial = reinterpret_cast<float*>(workspace.ptr + 4);
    unsigned* counter = workspace.ptr + 4 + std::size_t(cute_kernel::kGemvMaxSplits) * m;
    std::vector<std::pair<std::string, Launch>> launches = {
        {"gemv", [&] { CUDA_CHECK(cute_kernel::gemv(pa, px, py, m, k, stream)); }},
        {"gemv+workspace", [&] { CUDA_CHECK(cute_kernel::gemv(pa, px, py, m, k, stream, workspace.ptr + 4)); }}};
    if (k) {
        auto add = [&](std::string name, Launch fn) {
            launches.emplace_back(name, [=] { fn(); CUDA_CHECK(cudaGetLastError()); });
        };
        add("4/256", [=] { cute_kernel::gemv_launch<T, 4, 256>(pa, px, py, m, k, vec, stream); });
        add("8/256", [=] { cute_kernel::gemv_launch<T, 8, 256>(pa, px, py, m, k, vec, stream); });
        add("16/256", [=] { cute_kernel::gemv_launch<T, 16, 256>(pa, px, py, m, k, vec, stream); });
        add("32/256", [=] { cute_kernel::gemv_launch<T, 32, 256>(pa, px, py, m, k, vec, stream); });
        add("64/64", [=] { cute_kernel::gemv_launch<T, 64, 64>(pa, px, py, m, k, vec, stream); });
        add("128/128", [=] { cute_kernel::gemv_launch<T, 128, 128>(pa, px, py, m, k, vec, stream); });
        add("256/256", [=] { cute_kernel::gemv_launch<T, 256, 256>(pa, px, py, m, k, vec, stream); });
        add("512/512", [=] { cute_kernel::gemv_launch<T, 512, 512>(pa, px, py, m, k, vec, stream); });
        // 显式 Split-K：包括 split 数多于加载次数（部分 CTA 区间为空）的情况。
        for (int splits : {2, 3, 16}) {
            add("64/64,splits=" + std::to_string(splits), [=] {
                cute_kernel::gemv_launch<T, 64, 64>(pa, px, py, m, k, vec, stream, splits, partial, counter); });
            add("512/512,splits=" + std::to_string(splits), [=] {
                cute_kernel::gemv_launch<T, 512, 512>(pa, px, py, m, k, vec, stream, splits, partial, counter); });
        }
    }
    for (auto& [name, fn] : launches) {
        dy.upload(std::vector<T>(m + 2, T(-13)));
        fn(); CUDA_CHECK(cudaStreamSynchronize(stream));
        auto result = dy.download();
        require(bench::as_float(result.front()) == -13 && bench::as_float(result.back()) == -13, "GEMV output guard modified");
        std::vector<T> actual(result.begin() + 1, result.end() - 1);
        bool half = std::is_same_v<T, __half>;
        require(bench::compare(actual, ref, half ? 0.002 : 2e-5, half ? 0.002 : 2e-4).passed,
            "GEMV mismatch: " + std::to_string(m) + "x" + std::to_string(k) + ", config=" + name);
        fn(); CUDA_CHECK(cudaStreamSynchronize(stream));
        auto repeat = dy.download();
        require(std::memcmp(result.data(), repeat.data(), result.size() * sizeof(T)) == 0, "GEMV nondeterministic");
        auto guard = workspace.download();
        for (std::size_t w = 0; w < 4; ++w)
            require(guard[w] == 0xdeadbeefu && guard[guard.size() - 1 - w] == 0xdeadbeefu, "GEMV workspace guard modified");
        for (int r = 0; r < m; ++r)
            require(guard[4 + std::size_t(cute_kernel::kGemvMaxSplits) * m + r] == 0, "GEMV split counter not reset");
        if (m == 17 && k == 33 && name == "gemv+workspace" && !offset && !mode) {
            bench::Options options; options.iterations = 3; options.samples = 2; options.warmup = 1;
            bench::measure(fn, stream, options);
            auto captured = dy.download();
            require(std::memcmp(result.data(), captured.data(), result.size() * sizeof(T)) == 0, "GEMV graph replay mismatch");
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        host_tests();
        if (argc == 2 && std::string(argv[1]) == "--host-only") return 0;
        int devices = 0;
        auto status = cudaGetDeviceCount(&devices);
        if (status != cudaSuccess || !devices) {
            std::cout << "SKIP: GPU execution unavailable: " << cudaGetErrorString(status) << '\n'; return 77;
        }
        bench::Context context(0);
        require(context.prop.major >= 8, "GEMM tests require SM80+");
        std::vector<std::array<int, 3>> shapes = {
            {1, 1, 0}, {1, 1, 1}, {7, 9, 16}, {31, 33, 31}, {32, 32, 32},
            {63, 65, 33}, {65, 129, 40}, {65, 129, 97}, {128, 64, 64},
            {128, 128, 160}, {17, 35, 256}, {5, 7, 4097}, {4, 8, 8192}};
        for (auto s : shapes) for (int splits : {1, 2, 3, 5, 8}) {
            if (s[2] && splits > (s[2] + 31) / 32) continue;
            test_gemm(context.stream, s[0], s[1], s[2], splits, 0, 0);
        }
        for (int mode : {0, 1, 2}) {
            test_gemm(context.stream, 17, 35, 128, 3, 1, mode);
            test_gemm(context.stream, 17, 35, 128, 3, 0, mode);
        }
        std::cout << "PASS: GEMM FP64 references, tails, short pipelines, misalignment, guards, graph replay and determinism\n";
        for (auto s : std::vector<std::array<int, 2>>{{1, 0}, {1, 1}, {7, 8}, {17, 33}, {33, 256}, {129, 1024}, {3, 8193}, {5, 32768}, {9, 65536}}) {
            test_gemv<__half>(context.stream, s[0], s[1], 0, 0);
            test_gemv<float>(context.stream, s[0], s[1], 0, 0);
        }
        for (int mode : {0, 1, 2}) {
            test_gemv<__half>(context.stream, 17, 256, 1, mode);
            test_gemv<float>(context.stream, 17, 256, 1, mode);
        }
        std::cout << "PASS: GEMV FP16/FP32, all dispatch instances, FP64 references, tails, misalignment, guards, graph replay and determinism\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
