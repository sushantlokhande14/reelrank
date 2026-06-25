"""Format evaluation results as a Markdown table (reused in the README)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

METRICS = ("recall", "ndcg", "map")


def metrics_table(
    results: Mapping[str, Mapping[str, float]],
    k_values: Sequence[int],
    metrics: Sequence[str] = METRICS,
) -> str:
    """Render {model_name: metrics_dict} as a Markdown comparison table."""
    cols = ["model"] + [f"{m}@{k}" for k in k_values for m in metrics]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for name, mt in results.items():
        row = [name]
        for k in k_values:
            for m in metrics:
                row.append(f"{mt.get(f'{m}@{k}', 0.0):.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
