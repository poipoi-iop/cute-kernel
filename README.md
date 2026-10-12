以下内容由 AI 自动总结

## 独立 FA2 对比

```bash
.venv/bin/python bench/fa2_bench.py --output bench/reports/fa2
```

只编译 `bench/fa2.cu`，比较 `include/fa2` 与 `flash_attn.flash_attn_func`，不调用现有 `build.py` / `run.py`，也不运行 GEMM、GEMV 或 normalization benchmark。需要 CUDA、PyTorch 和 flash_attn；本机 `.venv` 已安装依赖。独立结果见 [bench/reports/fa2/report.md](bench/reports/fa2/report.md)，实现分析与计时口径见 [docs/fa2.md](docs/fa2.md)。

## 构建与运行

需要 CUDA Toolkit（本机为 12.8）、C++17 编译器、Python 3，以及仓库已有的 CuTe/CUTLASS 头文件。构建和绘图不需要 CMake、NumPy、Matplotlib 或 PyTorch。PyTorch 仅用于 softmax/RMSNorm 的可选工业基线。

在仓库根目录执行：

```bash
python3 bench/build.py --arch 80
bench/build/test_kernels --host-only
python3 -m unittest discover -s bench -p test_framework.py -v

# 以下命令需要实际 GPU。
bench/build/test_kernels
python3 bench/run.py --quick --output bench/results/quick
python3 bench/run.py --output bench/results/full
```

原始 hgemm/softmax 的编译错误保存在 build 日志中，不阻止新增 kernel 的构建。默认构建包括 SM80 cubin 和 compute_80 PTX；指定目标 GPU 时可用 `--arch 80,90`。原始代码的错误修复前，完整 suite 返回失败状态是预期现象，已有有效数据仍会保存。

```bash
# 可选：已安装支持当前 GPU 的 PyTorch 时，增加 Inductor 编译基线。
python3 bench/run.py --torch-compile --output bench/results/inductor

# 用普通 CUDA events 计时，独立输出，避免与 graph 数据混合。
python3 bench/run.py --timing events --output bench/results/events

# 只测 GEMV（GEMV 无调参，--no-tune 只影响 GEMM）。
python3 bench/run.py --ops gemv --output bench/results/gemv

# 自定义 shape 文件与重新生成图表。
python3 bench/run.py --shapes bench/shapes.json --output bench/results/custom
python3 bench/plot.py bench/results/custom
```

`run.py` 的退出码：0 表示本次全部已执行条目通过；1 表示构建缺失、编译/正确性/运行失败；77 表示没有 GPU。可选依赖缺失和特定配置不支持会保留记录。`test_kernels` 没有 GPU 时也返回 77，不能当作 GPU 测试通过。

建议在正式采集前运行真实 GPU 检查：

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 bench/build/test_kernels
compute-sanitizer --tool racecheck --error-exitcode 1 bench/build/test_kernels
compute-sanitizer --tool synccheck --error-exitcode 1 bench/build/test_kernels
```

## hgemm_splitk 的接口与实现

数学语义是 `C[M,N] = A[M,K] × B[N,K]^T`，三个矩阵均为连续行主序。输入、输出类型为 `cute::half_t`，MMA 累加、Split-K partial 和最终归约使用 FP32。与原 hgemm 的 FP16 累加精度不同，报告会标明这一点。

新实现沿用原文件的 CuTe tiled MMA、swizzled shared memory、`cp.async` 和 `ldmatrix` 结构。默认 CTA tile 为 `64×64×32`、128 线程、3 个 shared-memory stage。每个 `blockIdx.z` 分区计算独立 K 区间，完整写入自己的 FP32 partial。第二个 kernel 按固定顺序求和，最后转 FP16；无 atomic、无跨 CTA 自旋、无需清零 workspace。`splits=1` 直接写 C。

计划创建与 launch 分开，使 launch 不做分配、属性查询或同步，适合非默认 stream 和 CUDA Graph。以下片段省略调用方已有的 CUDA 错误处理：

```cpp
#include "hgemm_splitk.cuh"

cute_kernel::HgemmSplitKPlan plan;
// split=0 是启发式；sm_count 从调用方的 cudaDeviceProp 取得。
cudaError_t status = cute_kernel::make_hgemm_splitk_plan(
    &plan, M, N, K, /*split=*/0, sm_count);
// 检查 status 后，为 plan.workspace_bytes 分配 workspace；0 时使用 nullptr。
status = cute_kernel::launch_hgemm_splitk(
    plan, C, A, B, workspace, workspace_bytes, stream);
```

workspace 为 `4 × splits × M × N` 字节，`splits=1` 或空算子不需要 workspace。支持任意非负 M/N/K，包括不整齐的维度；`K=0` 清零输出，M/N 为 0 时不启动 kernel。非空 K 的显式 split 数不超过 `ceil(K/32)` 和 CUDA grid.z 上限。16-byte 对齐且 K 为 8 的倍数时走向量异步复制，否则走带边界检查的 scalar load。M/N 越界块也会做 predicate。

`SplitKConfig` 的模板内核可用于进一步开发 tile；当前稳定的 plan/launch 接口固定使用默认配置。默认启发式不是针对某一设备实测调出的最优参数。

## GEMV 的接口与优化

数学语义是连续行主序 `y[M] = A[M,K] × x[K]`，支持 `__half` 与 `float`，输出与输入类型相同，内部均为 FP32 FMA。实现为纯 CUDA（不依赖 CuTe），只有一个 kernel 模板 `gemv_kernel<T, G, THREADS, VEC>` 和一个 host 函数 `gemv()`。设计思路与 RTX 3090 实测数据见 [docs/gemv.md](docs/gemv.md)。

- 每 G 个相邻线程算一行：G≤32 时一个 warp 放多行，只用 shuffle 归约；G>32 时一个 CTA 算一行，warp 间经 shared memory 归约。
- 16 字节向量加载（8 个 FP16 / 4 个 FP32），每线程 4 路独立 load 并行在途；不满足对齐时走逐元素加载。
- `gemv()` 按"每线程约 4 次加载"选 G，无运行时调参。
- 行数少于 SM 数且单行 ≥ 64 KB 时，可选的 workspace 打开单 kernel Split-K：最后到达的 CTA 按固定顺序归约，结果确定，无第二次 launch。

```cpp
#include "gemv.cuh"

// 可选：Split-K workspace，分配后清零一次即可反复使用。
void* workspace;
cudaMalloc(&workspace, cute_kernel::gemv_workspace_bytes(M));
cudaMemset(workspace, 0, cute_kernel::gemv_workspace_bytes(M));
cudaError_t status = cute_kernel::gemv(A, x, y, M, K, stream, workspace);
```

workspace 为 `68 × M` 字节（16 个 FP32 partial + 1 个计数器每行），传 `nullptr` 则不走 Split-K。同一 workspace 不可被不同 stream 并发使用。

两类接口都要求调用方提供足够大的 device buffer，输入、输出及 workspace 不重叠，满足各元素的自然对齐，且在 stream 完成前保持有效。同一 workspace 不可被不同 stream 的并发 launch 共享。不同 K 分区可能改变浮点求和顺序；保证固定配置的归约顺序，而不保证与 cuBLAS 或其他 split 数逐 bit 相同。

新增源码统一使用四空格缩进。kernel 内注释说明线程/元素映射、K 分区、predicate、流水线同步与 workspace 生命周期。

## 评测覆盖与工业实现

默认 [shapes.json](shapes.json) 包含 15 个 GEMM shape、16 个 GEMV shape（分别测 FP16/FP32），以及 softmax、online softmax、RMSNorm 各 11 个 shape，共 80 个算子/精度/shape 组合。覆盖小 M/N 大 K、方阵、长条矩阵、K 短于流水线、尾部与不对齐尺寸。形状列表可直接增删；原始 normalization kernel 的 COL 是模板参数，如需新增可执行宽度，同时扩展 `build.py` 的 WIDTHS 再构建。

| 算子 | 自有实现 | 工业基线 |
|---|---|---|
| GEMM FP16 | `hgemm_splitk`、固定 split=1、新增适配器调用原 `hgemm` | cuBLAS GemmEx、cuBLASLt、CUTLASS SM80 |
| GEMV FP16 | 新 GEMV | cuBLAS GemmEx（N=1）、cuBLASLt |
| GEMV FP32 | 新 GEMV | cuBLAS Sgemv、cuBLASLt |
| Softmax / online softmax | 原 kernel | PyTorch softmax，可选 Inductor |
| RMSNorm FP32、无 weight | 原 kernel，eps=1e-5 | PyTorch F.rms_norm，可选 Inductor |

自有 GEMM 搜索 splits={1,2,4,8,16,32} 中合法且 workspace 不超过 32 MiB 的候选。GEMV 直接使用 `gemv()` 的内置启发式，不做搜索。`--no-tune` 使用 GEMM 默认计划。每个候选在参与搜索前检查正确性；无效候选会产生失败记录，不会悄悄进入最终性能图。

## 计时、准确性与报告

默认 CUDA Graph 捕获 50 次完整算子执行；10 次预热、9 组独立采样，输出 median/min/p95 和原始 samples。每组采样包含一次 graph replay；计时对象包括 Split-K partial 和 reduction，排除构造计划、分配、传输、参考计算与调参。`--timing events` 用同样迭代数直接提交操作，可能受到 CPU 提交速度影响，必须与 graph 结果分开比较。

输入固定 seed，GEMM/GEMV 使用均匀量级约 [-0.25,0.25)，normalization 使用 [-1,1)；Python 工业基线和 C++ 原算子使用相同生成公式。单独的 correctness 程序另含零输入、较大幅值的抵消、零 K、短流水线、misalignment、所有 GEMV dispatch 实例与显式 Split-K、输出/workspace guard、Split-K 计数器归零、非默认 stream、Graph replay 以及重复执行一致性检查。

| 检查 | 参考与误差上限 `atol + rtol × abs(ref)` |
|---|---|
| GEMM benchmark 全输出 | cuBLAS FP32 累加输出；atol=0.01，rtol=0.002 |
| GEMM 参考审计 | CPU FP64：输出≤4096 时全量，否则 257 个分布位置加 4 个角；atol=0.002，rtol=0.002 |
| GEMM 专项正确性测试 | CPU FP64 全输出；atol=0.002，rtol=0.002 |
| GEMV FP16 | CPU FP64 全输出；atol=0.002，rtol=0.002 |
| GEMV FP32 | CPU FP64 全输出；atol=2e-5，rtol=2e-4 |
| Softmax / RMSNorm | CPU FP64 全输出；atol=1e-6，rtol=1e-4 |


输出目录中包括：

- `results.json` / `results.csv`：所有结果，包括 compile_error、correctness_failed、unsupported、no_gpu 和依赖缺失。
- `metadata.json` / `build_manifest.json`：环境、形状、计时参数、编译命令、源码 SHA-256、日志位置。
- `*_median_us.svg`、`*_tflops.svg` / `*_effective_gbps.svg`：只有通过正确性检查的实测点。
- `*_speedup.svg`：对应 shape 下最快的有效工业基线耗时 / 自有实现耗时；没有基线则不计算。
- `coverage.svg` / `report.html`：状态覆盖图及可筛选的离线报告。

