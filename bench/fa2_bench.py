#!/usr/bin/env python3
"""Standalone FP16 FA2 forward comparison; never builds/runs the other kernels."""
import argparse
import csv
import ctypes
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ATOL, RTOL = 0.002, 0.002


def build_library(args, arch):
    build = ROOT / "bench/build/fa2"
    build.mkdir(parents=True, exist_ok=True)
    library = build / "libfa2.so"
    command = [args.nvcc, "-std=c++17", "-O3", "-lineinfo", "--expt-relaxed-constexpr",
               f"-arch=sm_{arch}", "-shared", "-Xcompiler", "-fPIC",
               "-I", str(ROOT / "include"), str(ROOT / "bench/fa2.cu"), "-o", str(library)]
    sources = [ROOT / "bench/fa2.cu", Path(__file__), args.shapes]
    sources += sorted((ROOT / "include/fa2").iterdir())
    sources += sorted((ROOT / "include/cute").rglob("*.hpp"))
    sources += sorted((ROOT / "include/cutlass").rglob("*.h"))
    hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()}
    print(f"Building only bench/fa2.cu (sm_{arch}) ...", flush=True)
    result = subprocess.run(command, text=True, capture_output=True)
    log = result.stdout + result.stderr
    (args.output / "build.log").write_text(log)
    manifest = {"command": command, "returncode": result.returncode, "source_sha256": hashes,
                "nvcc_version": subprocess.check_output([args.nvcc, "--version"], text=True)}
    (args.output / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"FA2 compilation failed; see {args.output / 'build.log'}\n{log[-3000:]}")
    return ctypes.CDLL(str(library))


def launch_fa2(lib, q, k, v, out, stream):
    batch, seq_q, heads, head_dim = q.shape
    strides = (ctypes.c_int * 12)(*(s for t in (q, k, v, out) for s in t.stride()[:3]))
    status = lib.fa2_launch(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
                           batch, seq_q, k.shape[1], heads, head_dim, strides, stream.cuda_stream)
    if status:
        raise RuntimeError(f"fa2_launch CUDA error {status}")
    return out


def reference(q, k, v):
    """FP32 full-output reference, chunked to avoid materializing an S x S matrix."""
    import torch
    qf, kf, vf = (t.transpose(1, 2).float() for t in (q, k, v))
    out = torch.empty_like(q, dtype=torch.float32)
    for start in range(0, q.shape[1], 128):
        scores = qf[:, :, start:start + 128] @ kf.transpose(-1, -2)
        probabilities = torch.softmax(scores / math.sqrt(q.shape[-1]), dim=-1)
        out[:, start:start + 128] = (probabilities @ vf).transpose(1, 2)
    return out


def accuracy(out, ref):
    import torch
    error = (out.float() - ref).abs()
    scaled = error / (ATOL + RTOL * ref.abs())
    valid = bool(torch.isfinite(out).all() and torch.isfinite(ref).all() and (scaled <= 1).all())
    return dict(status="ok" if valid else "correctness_failed", checked_elements=out.numel(),
                max_abs_error=float(error.max()), max_scaled_error=float(scaled.max()),
                rmse=float(error.square().mean().sqrt()))


def check_kernel(lib, stream):
    """Exercise multiple KV tiles, uneven Q/K lengths, strides and softmax rescaling."""
    import torch
    from flash_attn import flash_attn_func
    records = []
    for dim in (32, 64, 128):
        for kind in ("zero_q", "rescale", "strided"):
            q = torch.randn((2, 128, 3, dim), device="cuda", dtype=torch.float16)
            k = torch.randn((2, 192, 3, dim), device="cuda", dtype=torch.float16)
            v = torch.randn_like(k)
            out = torch.empty_like(q)
            if kind == "zero_q":
                q.zero_()
            elif kind == "rescale":
                # Later KV blocks have much larger maxima: exercise O and sum correction.
                q.fill_(1)
                k[:, :64].fill_(-2)
                k[:, 64:128].fill_(0)
                k[:, 128:].fill_(2)
            else:
                q = q.transpose(1, 2).contiguous().transpose(1, 2)
                k = k.transpose(1, 2).contiguous().transpose(1, 2)
                v = v.transpose(1, 2).contiguous().transpose(1, 2)
                # Padded output stride differs from Q; guard the unwritten slots.
                storage = torch.full((2, 256, 3, dim), 17., device="cuda", dtype=torch.float16)
                out = storage[:, ::2]
            out.fill_(float("nan"))
            launch_fa2(lib, q, k, v, out, stream)
            ref = reference(q, k, v)
            record = dict(kind=kind, head_dim=dim, **accuracy(out, ref))
            library_out = flash_attn_func(q, k, v, dropout_p=0., causal=False)
            record["flash_attn"] = accuracy(library_out, ref)
            if kind == "strided" and not bool((storage[:, 1::2] == 17).all()):
                record["status"] = "guard_failed"
            records.append(record)
    stream.synchronize()
    return records


def measure(operation, ref, stream, args):
    import torch
    out = operation()
    result = accuracy(out, ref)
    if result["status"] != "ok":
        return result
    for _ in range(args.warmup):
        out = operation()
    stream.synchronize()
    graph = None
    if args.timing == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(args.iterations):
                out = operation()
        graph.replay()
        stream.synchronize()
        # Verify captured execution too, before accepting any latency.
        result = accuracy(out, ref)
        if result["status"] != "ok":
            return result
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(args.samples):
        start.record(stream)
        if graph is not None:
            graph.replay()
        else:
            for _ in range(args.iterations):
                out = operation()
        stop.record(stream)
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000 / args.iterations)
    result.update(accuracy(out, ref))
    if result["status"] == "ok":
        result.update(median_us=statistics.median(samples), min_us=min(samples),
                      p95_us=sorted(samples)[math.ceil(0.95 * len(samples)) - 1], samples_us=samples)
    return result


def save_results(args, rows, metadata):
    # Failed numerical values are null in standard JSON; failures never get timing fields.
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    rows, metadata = clean(rows), clean(metadata)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (args.output / "results.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fields)
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# FA2 vs flash_attn", "", f"GPU: {metadata['gpu']}; FP16, non-causal, dropout=0, MHA.",
             f"PyTorch {metadata['torch']}; flash_attn {metadata['flash_attn']}; CUDA {metadata['cuda']}.",
             f"Timing: {args.timing}, warmup={args.warmup}, iterations={args.iterations}, samples={args.samples}.",
             "Both providers allocate O through PyTorch; graph timing excludes allocation and Python submission.",
             "TFLOP/s counts QK and PV only (4*B*H*Sq*Sk*D), excludes softmax.",
             f"Full-output FP32 reference with TF32 disabled; atol={ATOL}, rtol={RTOL}.", "",
             "| B | Sq | Sk | H | D | FA2 us | flash_attn us | FA2 / library latency |", "|---|---|---|---|---|---|---|---|"]
    for index in range(0, len(rows), 2):
        pair = rows[index:index + 2]
        own, library = pair[0], pair[1] if len(pair) == 2 else {}
        shape = " | ".join(str(own[key]) for key in ("batch", "seq_q", "seq_kv", "heads", "head_dim"))
        a, b = own.get("median_us"), library.get("median_us")
        lines.append(f"| {shape} | {f'{a:.3f}' if a else own['status']} | "
                     f"{f'{b:.3f}' if b else library.get('status', 'missing')} | {f'{a / b:.3f}x' if a and b else '-'} |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=Path, default=ROOT / "bench/fa2_shapes.json")
    parser.add_argument("--output", type=Path, default=ROOT / "bench/results/fa2")
    parser.add_argument("--nvcc", default=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timing", choices=("graph", "events"), default="graph")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.samples < 1:
        parser.error("invalid timing settings")
    shapes = json.loads(args.shapes.read_text())
    for shape in shapes:
        if (len(shape) != 5 or any(type(x) is not int or x <= 0 for x in shape) or
                shape[1] % 64 or shape[2] % 64 or shape[4] not in (32, 64, 128) or
                max(shape[1] * shape[3] * shape[4], shape[2] * shape[3] * shape[4]) * shape[0] > 2147483647):
            parser.error(f"unsupported [B, Sq, Sk, H, D]: {shape}")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    import flash_attn
    from flash_attn import flash_attn_func
    if not torch.cuda.is_available():
        print("CUDA unavailable", file=sys.stderr)
        return 77
    torch.cuda.set_device(args.device)
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        parser.error("FA2 requires SM80+")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    lib = build_library(args, capability[0] * 10 + capability[1])
    lib.fa2_launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 + [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p]
    lib.fa2_launch.restype = ctypes.c_int
    lib.fa2_setup.argtypes, lib.fa2_setup.restype = [ctypes.c_int], ctypes.c_int
    for dim in (32, 64, 128):
        status = lib.fa2_setup(dim)
        if status:
            raise RuntimeError(f"fa2_setup({dim}) CUDA error {status}")
    metadata = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    gpu=torch.cuda.get_device_name(), device=args.device, capability=capability,
                    torch=torch.__version__, cuda=torch.version.cuda, flash_attn=flash_attn.__version__,
                    python=sys.version, command=sys.argv, shapes=shapes,
                    options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    fa2_config="64x64, 4 warps, one Q/K/V shared-memory buffer set",
                    semantics="FP16 O=softmax(QK^T/sqrt(D))V; non-causal, dropout=0, equal Q/KV heads",
                    timing_scope="complete forward device execution; flash_attn also writes its internal LSE; allocation/reference/H2D excluded in graph mode",
                    cache_policy="warm reuse of inputs and graph-private output allocations",
                    accuracy=dict(atol=ATOL, rtol=RTOL, reference="chunked FP32, TF32 disabled"))
    stream = torch.cuda.Stream()
    rows = []
    with torch.cuda.stream(stream):
        metadata["kernel_checks"] = check_kernel(lib, stream)
        save_results(args, rows, metadata)
        if any(r["status"] != "ok" or r["flash_attn"]["status"] != "ok" for r in metadata["kernel_checks"]):
            raise RuntimeError(f"FA2 correctness checks failed; see {args.output / 'metadata.json'}")
        print("9 FA2 correctness cases passed (D=32/64/128; zero Q, softmax rescaling, strides).", flush=True)
        if args.check_only:
            return 0
        print("B Sq Sk H D | FA2 us | flash_attn us | FA2/library", flush=True)
        for batch, seq_q, seq_kv, heads, dim in shapes:
            q = torch.randn((batch, seq_q, heads, dim), device="cuda", dtype=torch.float16)
            k = torch.randn((batch, seq_kv, heads, dim), device="cuda", dtype=torch.float16)
            v = torch.randn_like(k)
            ref = reference(q, k, v)
            shape = dict(batch=batch, seq_q=seq_q, seq_kv=seq_kv, heads=heads, head_dim=dim,
                         dtype="fp16", causal=False, dropout=0., timing=args.timing)
            # Allocation is outside graph replay for both implementations.
            def own():
                return launch_fa2(lib, q, k, v, torch.empty_like(q), stream)
            def library():
                return flash_attn_func(q, k, v, dropout_p=0., softmax_scale=1 / math.sqrt(dim), causal=False)
            pair = []
            for provider, operation in (("fa2", own), ("flash_attn", library)):
                result = measure(operation, ref, stream, args)
                row = dict(shape, provider=provider, **result)
                if row["status"] == "ok":
                    row["tflops"] = 4 * batch * heads * seq_q * seq_kv * dim / (row["median_us"] * 1e6)
                pair.append(row)
            rows.extend(pair)
            save_results(args, rows, metadata)
            if all(r["status"] == "ok" for r in pair):
                a, b = (r["median_us"] for r in pair)
                print(f"{batch} {seq_q} {seq_kv} {heads} {dim} | {a:.3f} | {b:.3f} | {a / b:.3f}x", flush=True)
            else:
                print(f"{shape}: {[r['status'] for r in pair]}", flush=True)
    print(f"Saved {len(rows)} independent records: {args.output / 'report.md'}", flush=True)
    return int(any(r["status"] != "ok" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
