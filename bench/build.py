#!/usr/bin/env python3
"""Build without CMake or Python packages. Original headers are never rewritten."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = [32, 128, 256, 512, 768, 1024, 2048, 4096, 8192, 16384]


def hashes():
    """记录输入源码，构建结束再核对，确保编译过程没有改写原 kernel。"""
    paths = [p for directory in ("include", "src") for p in (ROOT / directory).rglob("*") if p.is_file()]
    paths += [p for p in (ROOT / "bench").iterdir() if p.is_file() and p.suffix in {".cu", ".hpp", ".py", ".json"}]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def original_source(op):
    """仅生成 ABI 适配器；不复制修补原实现，原文件错误保留在编译日志中。"""
    if op == "hgemm":
        return '''#include "gemm.cuh"
extern "C" int legacy_launch(const void* a, const void* b, void* c,
                            int m, int n, int k, cudaStream_t stream) {
    // 原实现不含尾部 predicate；适配器只放行完整 tile 与足够长的流水线。
    using C = config::Config<cute::half_t>;
    if (m % 128 || n % 128 || k % 32 || k < 160) return cudaErrorNotSupported;
    auto e = cudaFuncSetAttribute(hgemm<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kShmSize);
    if (e != cudaSuccess) return e;
    hgemm<C><<<dim3(n / 128, m / 128), C::kThreadNum, C::kShmSize, stream>>>(c, a, b, m, n, k);
    return cudaGetLastError();
}
'''
    header = "rms_norm.cuh" if op == "rms_norm" else "softmax.cuh"
    cases = "\n".join(f"        case {n}: {op}<1, {n}><<<m, 256, 0, stream>>>(x, y); break;" for n in WIDTHS)
    return f'''#include "{header}"
extern "C" int legacy_launch(const void* a, const void*, void* c,
                            int m, int n, int, cudaStream_t stream) {{
    // 原代码使用 float4：行 stride 也必须保持 16-byte 对齐。
    if (m < 1 || n < 1 || n % 4) return cudaErrorNotSupported;
    auto* x = const_cast<float*>(static_cast<const float*>(a));
    auto* y = static_cast<float*>(c);
    switch (n) {{
{cases}
        default: return cudaErrorNotSupported;
    }}
    return cudaGetLastError();
}}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "bench/build")
    parser.add_argument("--arch", default="80", help="comma-separated SMs; highest also includes PTX")
    parser.add_argument("--nvcc", default=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")
    args = parser.parse_args()
    if not re.fullmatch(r"\d+(,\d+)*", args.arch):
        parser.error("--arch must be numeric SMs, e.g. 80,90")
    archs = sorted({int(a) for a in args.arch.split(",")})
    if min(archs) < 80:
        parser.error("hgemm_splitk requires SM80+")
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    before = hashes()
    manifest = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "architectures": archs, "source_sha256": before, "targets": {},
                "original_widths": WIDTHS}
    base = [args.nvcc, "-std=c++17", "-O3", "-lineinfo", "--expt-relaxed-constexpr",
            "-I", str(ROOT / "include"), "-I", str(ROOT / "bench")]
    for arch in archs:
        base += [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    base += [f"-gencode=arch=compute_{archs[-1]},code=compute_{archs[-1]}"]
    manifest["nvcc_version"] = subprocess.run([args.nvcc, "--version"], text=True, capture_output=True, check=True).stdout

    def compile_target(name, command, output):
        """独立保存每个目标的命令和日志，避免一个原 kernel 的错误遮蔽其他结果。"""
        print(f"Building {name} ...", flush=True)
        start = time.monotonic()
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = build / f"{name}.log"
        log.write_text(result.stdout)
        status = "compiled" if result.returncode == 0 else "compile_error"
        manifest["targets"][name] = {"status": status, "returncode": result.returncode,
                                        "elapsed_seconds": time.monotonic() - start,
                                        "command": command, "log": str(log), "output": str(output)}
        # A failed rebuild must never load a stale successful plugin.
        if result.returncode and output.exists():
            output.unlink()
        print(f"  {status}: {log}", flush=True)
        return result.returncode == 0

    objects, success = [], True
    for name in ("main", "gemm", "gemv", "legacy"):
        output = build / f"{name}.o"
        success &= compile_target(name, base + ["-c", str(ROOT / f"bench/{name}.cu"), "-o", str(output)], output)
        objects.append(str(output))
    if success:
        output = build / "cute_bench"
        success &= compile_target("cute_bench", [args.nvcc, *objects, "-lcublas", "-lcublasLt", "-ldl", "-o", str(output)], output)
    elif (build / "cute_bench").exists():
        (build / "cute_bench").unlink()
    test_output = build / "test_kernels"
    success &= compile_target("test_kernels", base + [str(ROOT / "bench/test_kernels.cu"), "-lcublas", "-o", str(test_output)], test_output)
    for op in ("hgemm", "softmax", "online_softmax", "rms_norm"):
        name = "original_" + op
        source = build / f"{name}.cu"
        source.write_text(original_source(op))
        output = build / f"lib{name}.so"
        compile_target(name, base + ["-shared", "-Xcompiler", "-fPIC", str(source), "-o", str(output)], output)
    after = hashes()
    manifest["source_files_unchanged"] = before == after
    (build / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if before != after:
        raise SystemExit("source changed during build; inspect build_manifest.json")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
