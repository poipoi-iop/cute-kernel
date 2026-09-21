#!/usr/bin/env python3
"""Optional native PyTorch / Inductor normalization baselines; separate process."""
import argparse
import json
import math
import statistics


def run(case, settings, provider):
    """使用与原生 runner 相同的输入公式、容差和采样口径评测 PyTorch。"""
    import torch
    import torch.nn.functional as functional
    row = dict(case, provider=provider, timing=settings["timing"], cache="warm_reuse")
    torch.cuda.set_device(settings["device"])
    stream = torch.cuda.Stream()
    # Same deterministic FP32 values as the native runner, generated on CPU.
    i = torch.arange(case["m"] * case["n"], dtype=torch.int64)
    h = ((i ^ settings["seed"]) * 747796405 + 2891336453) & 0xffffffff
    h = (((h >> ((h >> 28) + 4)) ^ h) * 277803737) & 0xffffffff
    h = (h >> 22) ^ h
    host = ((h & 65535).float() - 32768) / 32768
    host = host.reshape(case["m"], case["n"])
    ref = host.double()
    if case["op"] == "rms_norm":
        ref = ref / torch.sqrt((ref * ref).mean(-1, keepdim=True) + 1e-5)
        operation = lambda x: functional.rms_norm(x, (case["n"],), weight=None, eps=1e-5)
        config = "torch.nn.functional.rms_norm,no_weight,eps=1e-5"
    else:
        shifted = ref - ref.max(-1, keepdim=True).values
        exponentials = shifted.exp()
        ref = exponentials / exponentials.sum(-1, keepdim=True)
        operation = lambda x: torch.softmax(x, -1)
        config = "torch.softmax,dim=-1"
    if provider == "pytorch_compile":
        operation = torch.compile(operation, fullgraph=True, dynamic=False)
        config += ",Inductor"
    # 正确性、首次执行及可选 Inductor 编译都在 event 区间之外。
    # Graph 模式仅重放捕获好的算子；events 模式保留正常 Python 提交路径。
    with torch.cuda.stream(stream):
        x = host.to("cuda")
        out = operation(x)  # Compile and first use are outside timing.
        stream.synchronize()
        got = out.cpu().double()
        errors = (got - ref).abs()
        scaled = errors / (1e-6 + 1e-4 * ref.abs())
        valid = bool(torch.isfinite(got).all() and torch.isfinite(ref).all() and (scaled <= 1).all())
        row.update(config=config, torch_version=torch.__version__, checked_elements=got.numel(),
                    max_abs_error=float(errors.max()), max_scaled_error=float(scaled.max()),
                    rmse=float((errors * errors).mean().sqrt()))
        if not valid:
            row.update(status="correctness_failed", note="timing withheld")
            return row
        for _ in range(settings["warmup"]):
            out = operation(x)
        stream.synchronize()
        graph = None
        if settings["timing"] == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(settings["iterations"]):
                    out = operation(x)
            graph.replay()
            stream.synchronize()
        samples = []
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(settings["samples"]):
            start.record(stream)
            if graph:
                graph.replay()
            else:
                for _ in range(settings["iterations"]):
                    out = operation(x)
            stop.record(stream)
            stop.synchronize()
            samples.append(start.elapsed_time(stop) * 1000 / settings["iterations"])
    median = statistics.median(samples)
    if median <= 0:
        raise RuntimeError("timer resolution insufficient; increase iterations")
    row.update(status="ok", note="", median_us=median, min_us=min(samples),
                p95_us=sorted(samples)[math.ceil(0.95 * len(samples)) - 1], samples_us=samples,
                effective_gbps=(case["m"] * case["n"] * 8) / (median * 1000))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--provider", choices=("pytorch_eager", "pytorch_compile"), required=True)
    args = parser.parse_args()
    case, settings = json.loads(args.case), json.loads(args.settings)
    try:
        row = run(case, settings, args.provider)
    except Exception as error:
        row = dict(case, provider=args.provider, status="runtime_error", note=str(error), timing=settings["timing"], cache="warm_reuse")
    for key, value in list(row.items()):
        if isinstance(value, float) and not math.isfinite(value):
            row[key] = None
    print(json.dumps(row, allow_nan=False))
    return 0 if row["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
