#include "common.hpp"
#include "hgemm_splitk.cuh"
#include "gemv.cuh"
#include <array>
#include <cstring>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// 在 host 上执行 CuTe UniversalCopy，独立核对每个线程取得的 A/x 元素。
// 参考地址由行主序公式计算，coverage 要求每个输入矩阵元素恰好被覆盖一次。
// 这能在没有 GPU 时检查 thread/value layout、x 广播和非整齐 split 的映射；
// 它不替代 GPU 上的 shuffle、数值精度或 race 检查。
template <class T, int Group, int Warps, bool Vectorized>
void host_gemv_copy_layout() {
    using namespace cute;
    using Config = cute_kernel::gemv_detail::CopyConfig<T, Group, Warps, Vectorized>;
    constexpr int Rows = Config::kRows, V = Config::kValues;
    int m = Rows + 3;
    typename Config::TiledCopy tiled_copy;
    for (int vectors : {1, Group + 3, 2 * Group + 1}) {
        int k = vectors * V;
        auto a = bench::inputs<T>(std::size_t(m) * k, 41);
        auto x = bench::inputs<T>(k, 71);
        if constexpr (Vectorized) {
            require((reinterpret_cast<std::uintptr_t>(a.data()) |
                    reinterpret_cast<std::uintptr_t>(x.data())) % 16 == 0,
                "host test buffers must support 16-byte copies");
        }
        auto matrix = make_tensor(
            make_gmem_ptr(a.data()), make_shape(m, k), make_stride(std::int64_t(k), _1{}));
        auto vector = make_tensor(
            make_gmem_ptr(x.data()), make_shape(m, k), make_stride(_0{}, _1{}));
        for (int splits : {1, 3, 8}) {
            if (splits > k) continue;
            std::vector<int> coverage(a.size(), 0);
            for (int split = 0; split < splits; ++split) {
                int start = int(std::int64_t(vectors) * split / splits);
                int end = int(std::int64_t(vectors) * (split + 1) / splits);
                for (int row_base = 0; row_base < m; row_base += Rows) {
                    auto origin = make_coord(row_base, std::int64_t(start) * V);
                    auto tiles_a = local_tile(
                        domain_offset(origin, matrix), typename Config::Tile{}, make_coord(_0{}, _));
                    auto tiles_x = local_tile(
                        domain_offset(origin, vector), typename Config::Tile{}, make_coord(_0{}, _));
                    for (int tid = 0; tid < Config::kThreads; ++tid) {
                        int row = row_base + tid / Group, lane = tid % Group;
                        if (row >= m) continue;
                        auto coord = typename Config::ThreadLayout{}.get_flat_coord(tid);
                        require(get<0>(coord) == tid / Group && get<1>(coord) == lane,
                            "CuTe subgroup/row mapping mismatch");
                        auto thread_copy = tiled_copy.get_slice(tid);
                        auto source_a = thread_copy.partition_S(tiles_a);
                        auto source_x = thread_copy.partition_S(tiles_x);
                        auto registers_a = make_tensor<T>(shape(source_a(_, _, _, 0)));
                        auto registers_x = make_tensor<T>(shape(source_x(_, _, _, 0)));
                        for (int base = start, tile = 0; base < end; base += Group, ++tile) {
                            if (base + lane >= end) continue;
                            copy(tiled_copy, source_a(_, _, _, tile), registers_a);
                            copy(tiled_copy, source_x(_, _, _, tile), registers_x);
                            for (int i = 0; i < V; ++i) {
                                int col = (base + lane) * V + i;
                                std::size_t index = std::size_t(row) * k + col;
                                require(bench::as_float(registers_a(i)) == bench::as_float(a[index]),
                                    "CuTe matrix copy address mismatch");
                                require(bench::as_float(registers_x(i)) == bench::as_float(x[col]),
                                    "CuTe vector broadcast address mismatch");
                                ++coverage[index];
                            }
                        }
                    }
                }
            }
            require(std::all_of(coverage.begin(), coverage.end(), [](int count) { return count == 1; }),
                "CuTe Split-K copy has missing or duplicate elements");
        }
    }
}

template <int Group, int Warps>
void host_gemv_copy_layout_variants() {
    host_gemv_copy_layout<__half, Group, Warps, true>();
    host_gemv_copy_layout<__half, Group, Warps, false>();
    host_gemv_copy_layout<float, Group, Warps, true>();
    host_gemv_copy_layout<float, Group, Warps, false>();
}

// 这些契约测试不执行 CUDA kernel，因而无 GPU 时也能验证维度、workspace
// 与空操作路径；真正的数值、越界和 Graph 检查在后面的 device 测试中完成。
void host_tests() {
    using namespace cute_kernel;
    HgemmSplitKPlan p;
    require(make_hgemm_splitk_plan(nullptr, 1, 1, 1) == cudaErrorInvalidValue, "null plan");
    require(make_hgemm_splitk_plan(&p, -1, 4, 32) == cudaErrorInvalidValue, "negative M");
    require(make_hgemm_splitk_plan(&p, 4, 4, 32, 2) == cudaErrorInvalidValue, "too many splits");
    require(make_hgemm_splitk_plan(&p, 4, 4, 32, 65536) == cudaErrorInvalidValue, "grid.z bound");
    CUDA_CHECK(make_hgemm_splitk_plan(&p, 65, 129, 97, 3));
    require(p.workspace_bytes == std::size_t(65) * 129 * 3 * 4, "workspace size");
    alignas(16) cute::half_t h[8]{};
    float tiny[1];
    require(launch_hgemm_splitk(p, h, h, h, tiny, sizeof(tiny)) == cudaErrorInvalidValue, "small workspace rejected");
    auto corrupted = p; corrupted.workspace_bytes = 0;
    require(launch_hgemm_splitk(corrupted, h, h, h, tiny, sizeof(tiny)) == cudaErrorInvalidValue, "tampered plan rejected");
    CUDA_CHECK(make_hgemm_splitk_plan(&p, 0, 17, 123));
    CUDA_CHECK(launch_hgemm_splitk(p, nullptr, nullptr, nullptr));
    CUDA_CHECK(make_hgemm_splitk_plan(&p, 8, 8, 0));
    require(p.splits == 1 && p.workspace_bytes == 0, "zero K needs no workspace");
    require(make_hgemm_splitk_plan(&p, 2147483647, 2147483647, 32) == cudaErrorInvalidConfiguration, "oversized grid rejected");
    GemvPlan g;
    require(make_gemv_plan(&g, -1, 8) == cudaErrorInvalidValue, "negative gemv M");
    require(make_gemv_plan(&g, 4, 8, 80, 7) == cudaErrorInvalidValue, "invalid subgroup");
    require(make_gemv_plan(&g, 4, 8, 80, 8, 3) == cudaErrorInvalidValue, "invalid warps");
    require(make_gemv_plan(&g, 4, 8, 80, 8, 4, 9) == cudaErrorInvalidValue, "invalid splits");
    CUDA_CHECK(make_gemv_plan(&g, 17, 33, 80, 8, 4, 3));
    require(g.workspace_bytes == 17 * 3 * 4, "gemv workspace size");
    require(launch_gemv<float>(g, tiny, tiny, tiny) == cudaErrorInvalidValue, "gemv missing workspace");
    CUDA_CHECK(make_gemv_plan(&g, 0, 33));
    CUDA_CHECK(launch_gemv<float>(g, nullptr, nullptr, nullptr));
    std::cout << "PASS: host API validation and workspace sizing\n";
    host_gemv_copy_layout_variants<4, 4>();
    host_gemv_copy_layout_variants<4, 8>();
    host_gemv_copy_layout_variants<8, 4>();
    host_gemv_copy_layout_variants<8, 8>();
    host_gemv_copy_layout_variants<16, 4>();
    host_gemv_copy_layout_variants<16, 8>();
    host_gemv_copy_layout_variants<32, 4>();
    host_gemv_copy_layout_variants<32, 8>();
    std::cout << "PASS: 32 CuTe GEMV copy layouts, broadcast, tails and Split-K coverage\n";
}

// 随机输入外再检查全零与大幅度抵消，避免只在温和数据上验证浮点归约。
template <class T>
void pattern(std::vector<T>& x, int mode) {
    if (mode == 1) for (auto& v : x) v = T(0);
    if (mode == 2) for (std::size_t i = 0; i < x.size(); ++i) x[i] = T(i % 2 ? -64.0f : 64.0f);
}

// 输出前后各保留一个 sentinel，workspace 前后各保留四个 float；执行后
// 同时核对 FP64 参考、guard 和重复执行的逐 bit 一致性。offset=1 专门
// 触发非 16-byte 对齐路径，检查 fallback 不会误用向量 load。
void test_gemm(cudaStream_t stream, int m, int n, int k, int splits, int offset, int mode) {
    using T = cute::half_t;
    cute_kernel::HgemmSplitKPlan plan;
    CUDA_CHECK(cute_kernel::make_hgemm_splitk_plan(&plan, m, n, k, splits));
    auto a = bench::inputs<T>(std::size_t(m) * k + offset, 31);
    auto b = bench::inputs<T>(std::size_t(n) * k + offset, 73);
    pattern(a, mode);
    if (mode == 2) for (auto& v : b) v = T(1);
    bench::Buffer<T> da(a.size()), db(b.size()), dc(std::size_t(m) * n + 2);
    da.upload(a); db.upload(b);
    std::vector<T> initialized(dc.size, T(-13)); dc.upload(initialized);
    bench::Buffer<float> workspace(plan.workspace_bytes / 4 + 8);
    workspace.upload(std::vector<float>(workspace.size, -1234));
    auto fn = [&] { CUDA_CHECK(cute_kernel::launch_hgemm_splitk(plan, dc.ptr + 1,
            da.ptr ? da.ptr + offset : nullptr, db.ptr ? db.ptr + offset : nullptr,
            workspace.ptr + 4, plan.workspace_bytes, stream)); };
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

// 枚举 group/warps/splits，覆盖 CuTe 不同 ThreadLayout/Copy_Atom 的实例，
// 并用行尾与 K 尾部 shape 验证 predicate；一个固定案例还检查 Graph 重放。
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
    for (int group : {4, 8, 16, 32}) for (int warps : {4, 8}) for (int splits : {1, 2, 3, 8}) {
        if (k && splits > k) continue;
        cute_kernel::GemvPlan plan;
        CUDA_CHECK(cute_kernel::make_gemv_plan(&plan, m, k, 80, group, warps, splits));
        bench::Buffer<float> workspace(plan.workspace_bytes / 4 + 8);
        workspace.upload(std::vector<float>(workspace.size, -1234));
        dy.upload(std::vector<T>(m + 2, T(-13)));
        auto fn = [&] { CUDA_CHECK(cute_kernel::launch_gemv(plan,
                da.ptr ? da.ptr + offset : nullptr, dx.ptr ? dx.ptr + offset : nullptr,
                dy.ptr + 1, workspace.ptr + 4, plan.workspace_bytes, stream)); };
        fn(); CUDA_CHECK(cudaStreamSynchronize(stream));
        auto result = dy.download();
        require(bench::as_float(result.front()) == -13 && bench::as_float(result.back()) == -13, "GEMV output guard modified");
        std::vector<T> actual(result.begin() + 1, result.end() - 1);
        bool half = std::is_same_v<T, __half>;
        require(bench::compare(actual, ref, half ? 0.002 : 2e-5, half ? 0.002 : 2e-4).passed,
            "GEMV mismatch: " + std::to_string(m) + "x" + std::to_string(k) + ", group=" + std::to_string(group));
        fn(); CUDA_CHECK(cudaStreamSynchronize(stream));
        auto repeat = dy.download();
        require(std::memcmp(result.data(), repeat.data(), result.size() * sizeof(T)) == 0, "GEMV nondeterministic");
        auto guard = workspace.download();
        for (std::size_t i = 0; i < guard.size(); ++i)
            if (i < 4 || i >= guard.size() - 4) require(guard[i] == -1234, "GEMV workspace guard modified");
        if (m == 17 && k == 33 && group == 8 && warps == 4 && splits == 3 && !offset && !mode) {
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
        for (auto s : std::vector<std::array<int, 2>>{{1, 0}, {1, 1}, {7, 8}, {17, 33}, {33, 256}, {129, 1024}, {3, 8193}}) {
            test_gemv<__half>(context.stream, s[0], s[1], 0, 0);
            test_gemv<float>(context.stream, s[0], s[1], 0, 0);
        }
        for (int mode : {0, 1, 2}) {
            test_gemv<__half>(context.stream, 17, 256, 1, mode);
            test_gemv<float>(context.stream, 17, 256, 1, mode);
        }
        std::cout << "PASS: GEMV FP16/FP32, all dispatches, FP64 references, tails, misalignment, guards, graph replay and determinism\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
