"""Paired comparison of one ReShapeBench method against the others, from metrics_per_edit.csv.

    python experiments/reshapebench_compare.py attn_q_f0   # -> outputs/reshapebench_eval/compare_attn_q_f0.md

Means per subset for every method, then the per-edit differences (method - other) with their standard error and
the share of edits where the method is better. Background metrics under the manual masks are added when
metrics_bg_manual_v1.json / manual_metrics_*.json hold a method's edits.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).resolve().parents[1] / "outputs/reshapebench_eval"
# name: (label, format, scale, higher is better)
METRICS = {"clip": ("CLIP-T", ".2f", 1, True), "clip_dir": ("CLIP-dir", ".4f", 1, True), "aesthetic": ("AS", ".3f", 1, True),
           "clip_i": ("CLIP-I", ".4f", 1, True), "dino": ("DINO", ".4f", 1, True), "bg_psnr": ("bg PSNR", ".2f", 1, True),
           "bg_lpips": ("bg LPIPS x1e3", ".1f", 1000, False)}
MANUAL = {"bg_psnr": ("bg PSNR (manual)", ".2f", 1, True), "bg_lpips": ("bg LPIPS x1e3 (manual)", ".1f", 1000, False)}


def table(title: str, methods: list[str], rows: dict, subsets: dict, metrics: dict) -> list[str]:
    lines = [f"### {title}", "", "| method | subset | n | " + " | ".join(label for label, *_ in metrics.values()) + " |",
             "|---|---|---|" + "---|" * len(metrics)]
    for method in methods:
        for subset, ids in subsets.items():
            cells = []
            for m, (_, fmt, scale, _) in metrics.items():
                values = [rows[method, i][m] for i in ids if rows[method, i][m] is not None]
                cells.append(format(np.mean(values) * scale, fmt))
            lines.append(f"| {method} | {subset} | {len(ids)} | " + " | ".join(cells) + " |")
    return lines + [""]


def paired(title: str, method: str, others: list[str], rows: dict, subsets: dict, metrics: dict) -> list[str]:
    lines = [f"### {title}", "", "Mean difference ± standard error (share of edits where " + method + " is better).", "",
             "| vs | subset | " + " | ".join(label for label, *_ in metrics.values()) + " |", "|---|---|" + "---|" * len(metrics)]
    for other in others:
        for subset, ids in subsets.items():
            cells = []
            for m, (_, fmt, scale, higher) in metrics.items():
                d = np.array([rows[method, i][m] - rows[other, i][m] for i in ids
                              if rows[method, i][m] is not None and rows[other, i][m] is not None]) * scale
                better = np.mean(d > 0 if higher else d < 0)
                cells.append(f"{d.mean():+{fmt}} ± {d.std(ddof=1) / np.sqrt(len(d)):{fmt}} ({better:.0%})")
            lines.append(f"| {other} | {subset} | " + " | ".join(cells) + " |")
    return lines + [""]


def main() -> None:
    method = sys.argv[1]
    released, subset_of = {}, {}
    for r in csv.DictReader(open(EVAL / "metrics_per_edit.csv", encoding="utf-8")):
        released[r["method"], r["id"]] = {m: float(r[m]) if r[m] not in ("", "None") else None for m in METRICS}
        subset_of[r["id"]] = r["subset"]
    methods = list(dict.fromkeys(m for m, _ in released))
    others = [m for m in methods if m != method]
    subsets = {"single": [i for i, s in subset_of.items() if s == "single"], "multi": [i for i, s in subset_of.items() if s == "multi"],
               "all": list(subset_of)}
    lines = [f"# {method} against the other ReShapeBench runs", ""]
    lines += table("Means, released masks", methods, released, subsets, METRICS)
    lines += paired(f"Paired differences, {method} − other, released masks", method, others, released, subsets, METRICS)

    manual = {}
    for file in (EVAL / "metrics_bg_manual_v1.json", *sorted(EVAL.glob("manual_metrics_*.json"))):
        if file.exists():
            for r in json.loads(file.read_text(encoding="utf-8")):
                m, i = r["key"].split("/")
                manual[m, i] = {k: r.get(k) for k in MANUAL}
    covered = [m for m in methods if all((m, i) in manual for i in subset_of)]
    if method in covered and len(covered) > 1:
        lines += table("Background, manual masks (v1, provisional)", covered, manual, subsets, MANUAL)
        lines += paired(f"Paired differences, {method} − other, manual masks", method, [m for m in covered if m != method],
                        manual, subsets, MANUAL)
    out = EVAL / f"compare_{method}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
