"""
Collect the per precision result files into one table and one plot.

    python -m src.report
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ORDER = ["fp16", "int8", "nf4"]
LABEL = {"fp16": "FP16", "int8": "INT8", "nf4": "NF4 4 bit"}


def load():
    rows = []
    for name in ORDER:
        path = RESULTS / f"{name}.json"
        if not path.exists():
            print(f"missing {path.name}, skipping")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {"precision": name}
        row.update(payload["metrics"])
        rows.append(row)
    return rows


def table(rows):
    header = (
        "| Precision | Overall | Reasoning | Coding | Tok/s | Mean latency | "
        "Peak memory |\n|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in rows:
        lines.append(
            "| {p} | {a:.0%} | {rr:.0%} | {c:.0%} | {t:.1f} | {l:.2f} s | {m:.0f} MiB |".format(
                p=LABEL[r["precision"]],
                a=r["accuracy_overall"],
                rr=r["accuracy_reasoning"],
                c=r["accuracy_coding"],
                t=r["tokens_per_second"],
                l=r["latency_mean_s"],
                m=r["peak_memory_mib"],
            )
        )
    return "\n".join(lines)


def plot(rows):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for r in rows:
        ax.scatter(r["peak_memory_mib"], r["accuracy_overall"] * 100, s=90)
        ax.annotate(
            LABEL[r["precision"]],
            (r["peak_memory_mib"], r["accuracy_overall"] * 100),
            textcoords="offset points",
            xytext=(8, 6),
        )
    ax.set_xlabel("Peak GPU memory (MiB)")
    ax.set_ylabel("Task accuracy (%)")
    ax.set_title("Quality against memory")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "pareto.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    rows = load()
    if not rows:
        raise SystemExit("no results yet, run src.run_eval first")
    md = table(rows)
    (RESULTS / "table.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    plot(rows)
