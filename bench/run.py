#!/usr/bin/env python3
"""Run shape suites, preserve failures, and generate CSV/JSON/SVG/HTML reports."""
import argparse
import csv
import datetime as dt
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OPS = ("gemm", "gemv", "softmax", "online_softmax", "rms_norm")


def cases_from_shapes(shapes, quick=False, ops=OPS):
    """把 shape 配置展开为单算子、单精度的 case；GEMV 同时覆盖 FP16/FP32。"""
    cases = []
    for op in ops:
        key = op if op in ("gemm", "gemv") else "normalization"
        entries = shapes[key]
        if quick:
            entries = [entries[i] for i in sorted({0, len(entries) // 2, len(entries) - 1})]
        for shape in entries:
            if not all(isinstance(x, int) and not isinstance(x, bool) and 0 < x <= 2147483647 for x in shape):
                raise ValueError(f"invalid shape: {shape}")
            if len(shape) != (3 if op == "gemm" else 2):
                raise ValueError(f"wrong rank for {op}: {shape}")
            m, n, k = shape if op == "gemm" else (shape[0], 1, shape[1]) if op == "gemv" else (shape[0], shape[1], 0)
            for dtype in (("fp16", "fp32") if op == "gemv" else ("fp16",) if op == "gemm" else ("fp32",)):
                cases.append(dict(op=op, dtype=dtype, m=m, n=n, k=k))
    return cases


def providers(case, torch_compile=False):
    """列出应该出现的 provider，使未执行或缺依赖的实现也能保留状态记录。"""
    if case["op"] == "gemm":
        return ["hgemm_splitk", "hgemm_splitk_s1", "cublas", "cublaslt", "cutlass_sm80", "original_hgemm"]
    if case["op"] == "gemv":
        return ["gemv", "cublas", "cublaslt"]
    return ["original_" + case["op"], "pytorch_eager"] + (["pytorch_compile"] if torch_compile else [])


def placeholder(case, provider, status, note, timing):
    """未测条目省略时间字段，不能用 0 延迟替代缺失数据。"""
    return dict(case, provider=provider, status=status, note=note, timing=timing, cache="warm_reuse")


def parse_output(text):
    """只接收原生 runner 的 JSON Lines，普通库诊断仍保存在原始日志中。"""
    rows = []
    for line in text.splitlines():
        if line.startswith("{"):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("benchmark output must contain JSON objects")
            rows.append(row)
    return rows


def build_status(row, manifest):
    """原文件编译失败优先于运行时不可用，保留可以定位到源码的根因。"""
    target = manifest.get("targets", {}).get(row["provider"], {})
    if row["provider"].startswith("original_") and target.get("status") == "compile_error":
        row.update(status="compile_error", note="Original header did not compile; see " + target.get("log", "build log"))
    return row


def write_results(destination, rows, metadata):
    """同时保存机器可读数据与 CSV；allow_nan=False 防止非标准 JSON 混入报告。"""
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    preferred = ["op", "dtype", "m", "n", "k", "provider", "status", "median_us", "min_us", "p95_us",
                    "tflops", "effective_gbps", "workspace_bytes", "max_abs_error", "max_scaled_error",
                    "rmse", "checked_elements", "config", "timing", "cache", "note", "samples_us"]
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    fields = [key for key in preferred if key in keys] + sorted(keys - set(preferred))
    with (destination / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "bench/build")
    parser.add_argument("--output", type=Path, default=ROOT / "bench/results")
    parser.add_argument("--shapes", type=Path, default=ROOT / "bench/shapes.json")
    parser.add_argument("--ops", default=",".join(OPS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--timing", choices=("graph", "events"), default="graph")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--torch-compile", action="store_true", help="also benchmark torch.compile/Inductor")
    parser.add_argument("--timeout", type=int, default=600, help="per-case limit in seconds")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.samples < 1 or args.timeout < 1:
        parser.error("invalid iteration or timeout settings")
    ops = args.ops.split(",")
    if not set(ops) <= set(OPS):
        parser.error("unknown operator in --ops")
    shapes = json.loads(args.shapes.read_text())
    cases = cases_from_shapes(shapes, args.quick, ops)
    build, output = args.build_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = build / "build_manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    executable = build / "cute_bench"
    gpu = {"status": "build_missing", "note": "run python3 bench/build.py first"}
    if executable.exists():
        probe = subprocess.run([str(executable), "--device-info", "--device", str(args.device)], text=True, capture_output=True, timeout=60)
        probe_rows = parse_output(probe.stdout)
        gpu = probe_rows[-1] if probe_rows else {"status": "runtime_error", "note": probe.stderr or probe.stdout}
    torch_available = importlib.util.find_spec("torch") is not None
    metadata = {"schema_version": 1, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "platform": platform.platform(), "python": sys.version, "gpu": gpu,
                "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "cases": cases, "torch_installed": torch_available,
                "timing_scope": "device execution of full operation, including Split-K reduction; setup and copies excluded",
                "cache_policy": "warm_reuse: repeatedly use the same allocations; effective GB/s is logical traffic, not measured HBM traffic",
                "accuracy": {"gemm": "all outputs vs cuBLAS FP32 accumulate; reference independently spot-checked in CPU FP64",
                                "gemv": "all outputs vs CPU FP64", "normalization": "all outputs vs CPU FP64"},
                "performance_claim": "No SOTA claim; only valid measured cases can support a GPU/shape-specific comparison."}
    if manifest:
        (output / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    # 每个 case 使用子进程，隔离原始 kernel 出错后的 CUDA context。
    # 完成一个 case 就持久化，后续 timeout/中断不会丢失已完成的数据。
    for index, case in enumerate(cases):
        expected = providers(case, args.torch_compile)
        if gpu.get("status") != "ok":
            for provider in expected:
                row = placeholder(case, provider, gpu["status"], gpu.get("note", "GPU unavailable"), args.timing)
                rows.append(build_status(row, manifest))
            continue
        print(f"[{index + 1}/{len(cases)}] {case}", flush=True)
        command = [str(executable), "--legacy-dir", str(build)]
        for key, value in case.items():
            command += ["--" + key, str(value)]
        for key in ("device", "timing", "warmup", "iterations", "samples", "seed"):
            command += ["--" + key, str(getattr(args, key))]
        if args.no_tune:
            command.append("--no-tune")
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout)
            current = parse_output(completed.stdout)
            (output / f"case_{index:03d}.log").write_text(completed.stdout + completed.stderr)
            for row in current:
                rows.append(build_status(row, manifest))
            got = {r["provider"] for r in current}
            for provider in expected:
                if provider not in got and not provider.startswith("pytorch_"):
                    rows.append(build_status(placeholder(case, provider, "runtime_error", "case did not emit provider; see case log", args.timing), manifest))
        except subprocess.TimeoutExpired:
            for provider in expected:
                if not provider.startswith("pytorch_"):
                    rows.append(placeholder(case, provider, "timeout", "per-case timeout exceeded", args.timing))
        if case["op"] not in ("gemm", "gemv"):
            for provider in (p for p in expected if p.startswith("pytorch_")):
                if not torch_available:
                    rows.append(placeholder(case, provider, "dependency_missing", "PyTorch is optional and is not installed", args.timing))
                    continue
                command = [sys.executable, str(ROOT / "bench/torch_baselines.py"), "--case", json.dumps(case),
                            "--settings", json.dumps({k: getattr(args, k) for k in ("device", "timing", "warmup", "iterations", "samples", "seed")}),
                            "--provider", provider]
                try:
                    completed = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout)
                    (output / f"case_{index:03d}_{provider}.log").write_text(completed.stdout + completed.stderr)
                    result = parse_output(completed.stdout)
                    rows.extend(result or [placeholder(case, provider, "runtime_error", completed.stderr[-2000:], args.timing)])
                except subprocess.TimeoutExpired:
                    rows.append(placeholder(case, provider, "timeout", "PyTorch compile/execution timeout exceeded", args.timing))
        # Preserve completed work if the suite is interrupted later.
        write_results(output, rows, metadata)
    write_results(output, rows, metadata)
    from plot import generate
    generate(output)
    valid = sum(row["status"] == "ok" for row in rows)
    print(f"Saved {len(rows)} records, {valid} valid timings: {output / 'report.html'}")
    if gpu.get("status") == "no_gpu":
        return 77
    if gpu.get("status") != "ok":
        return 1
    return int(any(r["status"] in {"correctness_failed", "runtime_error", "timeout", "compile_error"} for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
