#!/usr/bin/env python3
"""Dependency-free SVG charts and an offline HTML report; never invent timings."""
import argparse
from collections import Counter, defaultdict
from html import escape
import json
import math
from pathlib import Path

COLORS = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706", "#0891b2", "#64748b", "#be185d"]
INDUSTRIAL = {"cublas", "cublaslt", "cutlass_sm80", "pytorch_eager", "pytorch_compile"}


def finite_positive(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def valid(row, metric):
    """只有正确性通过、计时和目标指标均为有限正数的记录才能进入性能图。"""
    return row.get("status") == "ok" and finite_positive(row.get("median_us")) and finite_positive(row.get(metric))


def shape_key(row):
    return (row["m"], row["n"], row["k"])


def shape_label(row):
    values = (row["m"], row["n"], row["k"]) if row["op"] == "gemm" else (row["m"], row["k"]) if row["op"] == "gemv" else (row["m"], row["n"])
    return "×".join(map(str, values))


def speedup_rows(rows):
    """在相同算子/精度/shape/计时/cache 条件下，选最快有效工业实现作比较基准。"""
    baselines = {}
    for row in rows:
        if row["provider"] in INDUSTRIAL and valid(row, "median_us"):
            key = (row["op"], row["dtype"], shape_key(row), row.get("timing"), row.get("cache"))
            baselines[key] = min(row["median_us"], baselines.get(key, float("inf")))
    output = []
    for row in rows:
        key = (row["op"], row["dtype"], shape_key(row), row.get("timing"), row.get("cache"))
        if row["provider"] not in INDUSTRIAL and valid(row, "median_us") and key in baselines:
            output.append(dict(row, speedup=baselines[key] / row["median_us"]))
    return output


def svg_start(width, height, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f'<title>{escape(title)}</title>', '<rect width="100%" height="100%" fill="#fff"/>',
            '<style>text{font-family:Arial,sans-serif;fill:#1e293b;font-size:12px}.title{font-size:21px;font-weight:600}.sub{fill:#64748b}</style>',
            f'<text x="32" y="36" class="title">{escape(title)}</text>']


def chart(rows, metric, title, unit, destination):
    """导出独立 SVG；缺失结果保留空缺，不插值、不填零、不生成估计性能。"""
    shapes = list(dict.fromkeys(shape_key(row) for row in rows))
    usable = [row for row in rows if valid(row, metric)]
    width, height = max(1000, 160 + 67 * len(shapes)), 650
    parts = svg_start(width, height, title)
    if not usable:
        statuses = Counter(row.get("status", "unknown") for row in rows)
        parts += ['<text x="40" y="110" style="font-size:23px">No valid measurements</text>',
                    '<text x="40" y="145" class="sub">No latency, throughput, or speedup values have been synthesized.</text>']
        for i, (status, count) in enumerate(sorted(statuses.items())):
            parts.append(f'<text x="40" y="{195 + i * 26}">{escape(status)}: {count} records</text>')
        parts.append('</svg>')
        destination.write_text("\n".join(parts))
        return
    left, right, top, bottom = 85, width - 45, 135, 470
    values = [r[metric] for r in usable]
    low, high = math.floor(math.log10(min(values))), math.ceil(math.log10(max(values)))
    if low == high:
        low -= 1; high += 1
    if metric == "speedup":
        low, high = min(low, -1), max(high, 1)
    def y(value):
        return bottom - (math.log10(value) - low) / (high - low) * (bottom - top)
    def x(index):
        return left + (index + 0.5) / max(1, len(shapes)) * (right - left)
    parts.append(f'<text x="32" y="62" class="sub">{escape(unit)} · log scale · only correctness-passing results · warm allocation reuse</text>')
    for power in range(low, high + 1):
        pos, value = y(10 ** power), 10 ** power
        parts.append(f'<line x1="{left}" x2="{right}" y1="{pos:.1f}" y2="{pos:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left - 9}" y="{pos + 4:.1f}" text-anchor="end">{value:g}</text>')
    labels = {shape_key(row): shape_label(row) for row in rows}
    for i, shape in enumerate(shapes):
        parts.append(f'<text transform="translate({x(i):.1f},{bottom + 16}) rotate(50)" text-anchor="start">{escape(labels[shape])}</text>')
    providers = sorted({r["provider"] for r in usable})
    for index, provider in enumerate(providers):
        color = COLORS[index % len(COLORS)]
        px, py = 35 + (index % 4) * ((width - 70) / 4), 88 + (index // 4) * 20
        parts.append(f'<circle cx="{px}" cy="{py}" r="4" fill="{color}"/><text x="{px + 10}" y="{py + 4}">{escape(provider)}</text>')
        points = []
        for row in usable:
            if row["provider"] != provider:
                continue
            pos_x, pos_y = x(shapes.index(shape_key(row))), y(row[metric])
            points.append((shapes.index(shape_key(row)), pos_x, pos_y))
            tooltip = f'{provider}: {shape_label(row)}; {metric}={row[metric]:.5g}; {row.get("config", "")}'
            if metric == "median_us" and finite_positive(row.get("min_us")) and finite_positive(row.get("p95_us")):
                parts.append(f'<line x1="{pos_x:.1f}" x2="{pos_x:.1f}" y1="{y(row["min_us"]):.1f}" y2="{y(row["p95_us"]):.1f}" stroke="{color}" opacity="0.5"/>')
            parts.append(f'<circle cx="{pos_x:.1f}" cy="{pos_y:.1f}" r="4" fill="{color}"><title>{escape(tooltip)}</title></circle>')
        # Connect only adjacent supported shapes, preserving gaps for failures.
        points.sort()
        for (a, ax, ay), (b, bx, by) in zip(points, points[1:]):
            if b == a + 1:
                parts.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" opacity="0.35"/>')
    if metric == "speedup":
        parts.append(f'<line x1="{left}" x2="{right}" y1="{y(1):.1f}" y2="{y(1):.1f}" stroke="#334155" stroke-dasharray="5 4"/>')
    parts.append(f'<text x="32" y="625" class="sub">Shapes are categorical. Latency whiskers: minimum–p95. Speedup baseline: fastest valid industrial provider per shape.</text>')
    parts.append('</svg>')
    destination.write_text("\n".join(parts))


def coverage(rows, destination):
    """统计编译/正确性/环境状态的条目数；此图本身不表达吞吐或延迟。"""
    groups = defaultdict(Counter)
    for row in rows:
        groups[(row["op"], row["provider"])][row["status"]] += 1
    width, height = 1120, 125 + 30 * len(groups)
    parts = svg_start(width, height, "Benchmark coverage and validation status")
    parts.append('<text x="32" y="63" class="sub">Counts of records, not performance. Failed or unavailable entries have no timing values.</text>')
    colors = {"ok": "#059669", "compile_error": "#dc2626", "correctness_failed": "#dc2626", "no_gpu": "#64748b"}
    maximum = max((sum(c.values()) for c in groups.values()), default=1)
    for i, ((op, provider), counts) in enumerate(sorted(groups.items())):
        y = 92 + i * 30
        parts.append(f'<text x="32" y="{y + 14}">{escape(op + " / " + provider)}</text>')
        x = 350
        for status, count in sorted(counts.items()):
            length = count / maximum * 360
            parts.append(f'<rect x="{x}" y="{y}" width="{length:.1f}" height="19" fill="{colors.get(status, "#d97706")}"><title>{escape(status)}: {count}</title></rect>')
            x += length
        label = ", ".join(f"{status}: {count}" for status, count in sorted(counts.items()))
        parts.append(f'<text x="730" y="{y + 14}">{escape(label)}</text>')
    parts.append('</svg>')
    destination.write_text("\n".join(parts))


def generate(directory):
    """从已保存的数据重建图表，并拒绝把不同计时方法放在同一份比较报告中。"""
    directory = Path(directory)
    rows = json.loads((directory / "results.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text()) if (directory / "metadata.json").exists() else {}
    methods = {(r.get("timing"), r.get("cache")) for r in rows if r.get("status") == "ok"}
    if len(methods) > 1:
        raise ValueError("mixed timing/cache policies; use separate reports for fair comparisons")
    coverage(rows, directory / "coverage.svg")
    images = ["coverage.svg"]
    for op, dtype in sorted({(r["op"], r["dtype"]) for r in rows}):
        subset = [r for r in rows if r["op"] == op and r["dtype"] == dtype]
        for metric, unit in (("median_us", "Latency (microseconds); lower is better"),
                                ("tflops", "TFLOPS; higher is better") if op == "gemm" else ("effective_gbps", "Logical GB/s; higher is better")):
            name = f"{op}_{dtype}_{metric}.svg"
            chart(subset, metric, f"{op} · {dtype}", unit, directory / name)
            images.append(name)
        speedups = speedup_rows(subset)
        if speedups:
            name = f"{op}_{dtype}_speedup.svg"
            chart(speedups, "speedup", f"{op} · {dtype} · speedup", "Speedup vs fastest measured industrial baseline", directory / name)
            images.append(name)
    valid_count = sum(valid(r, "median_us") for r in rows)
    gpu = metadata.get("gpu", {})
    details = escape(gpu.get("gpu", gpu.get("note", "unknown GPU")))
    html = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>CUDA kernel benchmark</title>',
            '<style>body{font:15px system-ui,sans-serif;margin:32px;color:#172033}img{max-width:100%;border:1px solid #ddd;margin:12px 0}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:6px;text-align:left}th{position:sticky;top:0;background:#eef2ff}td{font-size:12px}.note{padding:16px;background:#f1f5f9}input{padding:8px;margin:12px 0;width:350px}</style>',
            '<h1>CUDA kernel benchmark</h1>',
            f'<p>{valid_count} valid timings / {len(rows)} records. Device: {details}</p>',
            '<p class="note">Only correctness-passing measurements enter performance charts. No GPU means no performance evidence and no SOTA claim. Timings include all Split-K reduction work. Reused buffers may fit in L2; logical GB/s is not measured HBM bandwidth. CUTLASS is one fixed SM80 configuration. FP16 outputs use FP32 accumulation in new kernels.</p>',
            '<p><a href="results.csv">CSV</a> · <a href="results.json">JSON</a> · <a href="metadata.json">Environment and methodology</a></p>']
    html.extend(f'<div><a href="{name}"><img src="{name}" alt="{name}"></a></div>' for name in images)
    html += ['<h2>All records</h2><input id="filter" placeholder="Filter by operator, provider, shape, or status">',
                '<table id="results"><thead><tr><th>Operator / dtype</th><th>Shape</th><th>Provider</th><th>Status</th><th>Median µs</th><th>p95 µs</th><th>Configuration / note</th></tr></thead><tbody>']
    for row in rows:
        def value(key):
            number = row.get(key)
            return f"{number:.5g}" if isinstance(number, (float, int)) and math.isfinite(number) else "—"
        cells = [row["op"] + " / " + row["dtype"], shape_label(row), row["provider"], row["status"],
                    value("median_us"), value("p95_us"), row.get("config", "") + " " + row.get("note", "")]
        html.append('<tr>' + ''.join(f'<td>{escape(str(cell))}</td>' for cell in cells) + '</tr>')
    html += ['</tbody></table>', '<script>document.getElementById("filter").addEventListener("input",function(){const q=this.value.toLowerCase();document.querySelectorAll("#results tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))});</script></html>']
    (directory / "report.html").write_text("\n".join(html))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    generate(parser.parse_args().directory)
