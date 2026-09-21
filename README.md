以下内容由 AI 自动总结

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

# 只测 GEMV，关闭搜索以评测默认启发式。
python3 bench/run.py --ops gemv --no-tune --output bench/results/gemv_default

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

数学语义是连续行主序 `y[M] = A[M,K] × x[K]`，支持 `__half` 与 `float`，输出与输入类型相同，内部均为 FP32 FMA。

- CuTe `ThreadLayout` 为 `(Rows,Group):(Group,1)`，`ValueLayout` 为 `(1,V)`，由 `make_tiled_copy` 明确分配每个线程的行与连续 K 元素。
- A 使用行主序 CuTe tensor；x 使用 M 维 stride=0 的广播视图。`domain_offset` 定位 Split-K 起点，`local_tile` 与 `partition_S` 提取线程分片，`cute::copy` 将数据加载到寄存器 tensor。
- 16-byte 加载：每线程 8 个 FP16 或 4 个 FP32；半精度先转换为 FP32 再相乘累加。
- 4 个独立累加器降低依赖链；同一行用 4/8/16/32 个 lane，CTA 为 4/8 个 warp。
- 多行并行和合并访问，warp subgroup shuffle 归约，无常规路径的 shared-memory barrier。
- M 小、K 大时可沿 K 增加 CTA 并行度，再归约 FP32 partial；包含完整 workspace 和尾部处理。
- 行 stride 或指针不满足向量对齐时走 scalar fallback；支持非整齐尺寸与 K=0。

```cpp
#include "gemv.cuh"

cute_kernel::GemvPlan plan;
cudaError_t status = cute_kernel::make_gemv_plan(
    &plan, M, K, sm_count, /*group=*/0, /*warps=*/4, /*splits=*/0);
// 检查 status 并按 plan.workspace_bytes 分配 workspace。
status = cute_kernel::launch_gemv(
    plan, A, x, y, workspace, workspace_bytes, stream);
```

FP32 partial workspace 为 `4 × splits × M` 字节，split=1 时为 0。`make_gemv_plan` 的 group=0、splits=0 使用默认启发式，bench 会测量候选并把胜出的 group/warps/splits 写进结果。候选搜索与最终计时分开；库接口本身不会偷偷调参。

两类接口都要求调用方提供足够大的 device buffer，输入、输出及 workspace 不重叠，满足各元素的自然对齐，且在 stream 完成前保持有效。同一 workspace 不可被不同 stream 的并发 launch 共享。不同 K 分区可能改变浮点求和顺序；保证固定配置的归约顺序，而不保证与 cuBLAS 或其他 split 数逐 bit 相同。

新增源码统一使用四空格缩进。kernel 内注释说明 CuTe layout、线程/元素映射、K 分区、predicate、流水线同步与 workspace 生命周期。`test_kernels --host-only` 还会实际执行 32 种 CuTe host copy 配置，核对广播、尾部以及 Split-K 对全部矩阵元素的唯一覆盖；这些检查不替代 GPU 数值与并发验证。

## 评测覆盖与工业实现

默认 [shapes.json](shapes.json) 包含 15 个 GEMM shape、16 个 GEMV shape（分别测 FP16/FP32），以及 softmax、online softmax、RMSNorm 各 11 个 shape，共 80 个算子/精度/shape 组合。覆盖小 M/N 大 K、方阵、长条矩阵、K 短于流水线、尾部与不对齐尺寸。形状列表可直接增删；原始 normalization kernel 的 COL 是模板参数，如需新增可执行宽度，同时扩展 `build.py` 的 WIDTHS 再构建。

| 算子 | 自有实现 | 工业基线 |
|---|---|---|
| GEMM FP16 | `hgemm_splitk`、固定 split=1、新增适配器调用原 `hgemm` | cuBLAS GemmEx、cuBLASLt、CUTLASS SM80 |
| GEMV FP16 | 新 GEMV | cuBLAS GemmEx（N=1）、cuBLASLt |
| GEMV FP32 | 新 GEMV | cuBLAS Sgemv、cuBLASLt |
| Softmax / online softmax | 原 kernel | PyTorch softmax，可选 Inductor |
| RMSNorm FP32、无 weight | 原 kernel，eps=1e-5 | PyTorch F.rms_norm，可选 Inductor |

自有 GEMM 搜索 splits={1,2,4,8,16,32} 中合法且 workspace 不超过 32 MiB 的候选。GEMV 搜索 group={4,8,16,32}、warps={4,8} 和适合 M/K 的 split 数。`--no-tune` 使用默认计划。每个候选在参与搜索前检查正确性；无效候选会产生失败记录，不会悄悄进入最终性能图。

## 计时、准确性与报告

默认 CUDA Graph 捕获 50 次完整算子执行；10 次预热、9 组独立采样，输出 median/min/p95 和原始 samples。每组采样包含一次 graph replay；计时对象包括 Split-K partial 和 reduction，排除构造计划、分配、传输、参考计算与调参。`--timing events` 用同样迭代数直接提交操作，可能受到 CPU 提交速度影响，必须与 graph 结果分开比较。

输入固定 seed，GEMM/GEMV 使用均匀量级约 [-0.25,0.25)，normalization 使用 [-1,1)；Python 工业基线和 C++ 原算子使用相同生成公式。单独的 correctness 程序另含零输入、较大幅值的抵消、零 K、短流水线、misalignment、所有 GEMV dispatch、输出/workspace guard、非默认 stream、Graph replay 以及重复执行一致性检查。

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

