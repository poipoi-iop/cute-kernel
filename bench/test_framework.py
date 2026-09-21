#!/usr/bin/env python3
"""Host-only checks for reporting honesty, fairness and result persistence."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import plot
import run


def fixture(provider, time, status="ok", **extra):
    """Synthetic values exist only in temporary test files, never in reports."""
    return dict(op="gemv", dtype="fp16", m=16, n=1, k=32, provider=provider,
                status=status, median_us=time, min_us=time, p95_us=time,
                timing="graph", cache="warm_reuse", effective_gbps=1, **extra)


class FrameworkTests(unittest.TestCase):
    def test_fastest_valid_baseline_only(self):
        rows = [fixture("gemv", 5), fixture("cublas", 20), fixture("cublaslt", 10),
                fixture("cutlass_sm80", 0.1, "correctness_failed")]
        result = plot.speedup_rows(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["speedup"], 2)

    def test_missing_baseline_is_not_zero_or_infinity(self):
        self.assertEqual(plot.speedup_rows([fixture("gemv", 5)]), [])
        self.assertFalse(plot.valid(fixture("gemv", 0), "median_us"))
        self.assertFalse(plot.valid(fixture("gemv", float("nan")), "median_us"))
        self.assertFalse(plot.valid(fixture("gemv", -1), "median_us"))

    def test_comparison_requires_same_shape_dtype_timing_cache(self):
        new = fixture("gemv", 5)
        for key, value in (("m", 17), ("dtype", "fp32"), ("timing", "events"), ("cache", "cold")):
            baseline = fixture("cublas", 10)
            baseline[key] = value
            self.assertEqual(plot.speedup_rows([new, baseline]), [])

    def test_no_gpu_report_contains_no_performance_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = dict(op="gemm", dtype="fp16", m=32, n=64, k=4096)
            rows = [run.placeholder(case, "hgemm_splitk", "no_gpu", "GPU unavailable", "graph")]
            run.write_results(root, rows, {"gpu": {"status": "no_gpu"}})
            plot.generate(root)
            image = root / "gemm_fp16_median_us.svg"
            self.assertIn("No valid measurements", image.read_text())
            ET.parse(image)
            self.assertNotIn("median_us", json.loads((root / "results.json").read_text())[0])
            self.assertFalse((root / "gemm_fp16_speedup.svg").exists())

    def test_failed_results_do_not_enter_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.svg"
            rows = [fixture("good", 10), fixture("incorrect_fast", 0.1, "correctness_failed")]
            plot.chart(rows, "median_us", "synthetic test", "us", path)
            text = path.read_text()
            self.assertNotIn("incorrect_fast", text)
            self.assertIn("good", text)
            ET.parse(path)

    def test_csv_keeps_invalid_timing_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unavailable = run.placeholder(dict(op="gemv", dtype="fp16", m=1, n=1, k=64), "gemv", "no_gpu", "unavailable", "graph")
            run.write_results(root, [fixture("cublas", 5), unavailable], {})
            with (root / "results.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[1]["median_us"], "")

    def test_compile_error_survives_gpu_unavailability(self):
        row = fixture("original_hgemm", 0, "no_gpu")
        run.build_status(row, {"targets": {"original_hgemm": {"status": "compile_error", "log": "actual.log"}}})
        self.assertEqual(row["status"], "compile_error")
        self.assertIn("actual.log", row["note"])

    def test_mixed_timing_reports_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            second = fixture("cublas", 10); second["timing"] = "events"
            run.write_results(root, [fixture("gemv", 5), second], {})
            with self.assertRaises(ValueError):
                plot.generate(root)

    def test_shape_expansion_and_validation(self):
        shapes = {"gemm": [[1, 2, 3]], "gemv": [[4, 5]], "normalization": [[6, 8]]}
        cases = run.cases_from_shapes(shapes)
        self.assertEqual(len(cases), 6)
        self.assertEqual({c["dtype"] for c in cases if c["op"] == "gemv"}, {"fp16", "fp32"})
        shapes["gemm"] = [[1, 0, 3]]
        with self.assertRaises(ValueError):
            run.cases_from_shapes(shapes)

    def test_strict_json_rejects_nonfinite_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run.write_results(Path(tmp), [fixture("gemv", float("nan"))], {})


if __name__ == "__main__":
    unittest.main()
