#Yu's code to generate the plots 
from __future__ import annotations

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import finchlite as fl
from finchlite import ffuncs as ffunc
from finchlite.autoschedule.tensor_stats import (
    BlockedStats,
    BlockedStatsFactory,
    DatabaseStatsFactory,
    DCStatsFactory,
    DenseStatsFactory,
    UniformStatsFactory,
)
from finchlite.finch_logic import Field

INNER_FACTORIES = [
    ("DenseStats", DenseStatsFactory()),
    ("DCStats", DCStatsFactory()),
    ("DatabaseStats", DatabaseStatsFactory()),
    ("UniformStats", UniformStatsFactory()),
]

MATRIX_TYPES = ["diagonal", "tridiagonal", "banded", "triangular", "striped"]
CSV_PATH = "benchmarks/stats_errors.csv"


def load_errors():
    if not os.path.exists(CSV_PATH):
        return None
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        errors = {}
        for row in reader:
            for name, val in row.items():
                if name == "matrix_type":
                    continue
                errors.setdefault(name, []).append(float(val))
    print(f"Loaded {CSV_PATH}")
    return errors


def save_errors(errors):
    stat_names = list(errors.keys())
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["matrix_type"] + stat_names)
        writer.writeheader()
        for idx, m_type in enumerate(MATRIX_TYPES):
            row = {"matrix_type": m_type}
            for name in stat_names:
                row[name] = errors[name][idx]
            writer.writerow(row)
    print(f"Saved {CSV_PATH}")


def get_structured_example(rows, cols, matrix_type):
    if matrix_type == "diagonal":
        return np.eye(rows, cols, dtype=np.float64)
    if matrix_type == "tridiagonal":
        A = np.eye(rows, cols, k=0) + np.eye(rows, cols, k=1) + np.eye(rows, cols, k=-1)
        return (A > 0).astype(np.float64)
    if matrix_type == "banded":
        bw = 5
        r, c = np.indices((rows, cols))
        return (np.abs(r - c) <= bw).astype(np.float64)
    if matrix_type == "triangular":
        return np.triu(np.ones((rows, cols), dtype=np.float64))
    if matrix_type == "striped":
        A = np.zeros((rows, cols), dtype=np.float64)
        A[:, ::5] = 1.0
        return A
    return np.zeros((rows, cols), dtype=np.float64)


def compute_errors():
    M, K, N = 50, 50, 50
    i, j, k = Field("i"), Field("j"), Field("k")
    blocks_per_dim = {i: 5, j: 5, k: 5}

    stats_configs = [(name, factory, False) for name, factory in INNER_FACTORIES] + [
        (f"Blocked{name}", factory, True) for name, factory in INNER_FACTORIES
    ]

    errors = {name: [] for name, _, _ in stats_configs}

    for m_type in MATRIX_TYPES:
        data_a = get_structured_example(M, K, m_type)
        data_b = get_structured_example(K, N, m_type)
        tns_a = fl.asarray(data_a)
        tns_b = fl.asarray(data_b)

        actual_nnz = float(np.count_nonzero(data_a.astype(bool) @ data_b.astype(bool)))

        for name, factory, blocked in stats_configs:
            if not blocked:
                g_a = factory(tns_a, (i, k))
                g_b = factory(tns_b, (k, j))
                g_res = factory.aggregate(
                    ffunc.add, 0.0, (k,), factory.mapjoin(ffunc.mul, g_a, g_b)
                )
                est = g_res.estimate_non_fill_values()
            else:
                bf = BlockedStatsFactory(blocks_per_dim, factory)
                b_a = BlockedStats.from_tensor(tns_a, (i, k), blocks_per_dim, factory)
                b_b = BlockedStats.from_tensor(tns_b, (k, j), blocks_per_dim, factory)
                b_res = bf.aggregate(
                    ffunc.add, 0.0, (k,), bf.mapjoin(ffunc.mul, b_a, b_b)
                )
                est = b_res.estimate_non_fill_values()

            errors[name].append((est - actual_nnz) / actual_nnz)

    return errors


def plot(errors):
    stats = list(errors.keys())
    n_stats = len(stats)
    n_types = len(MATRIX_TYPES)

    x = np.arange(n_stats)
    bar_width = 0.15

    _, ax = plt.subplots(figsize=(14, 8))

    simple_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]

    band_colors = plt.cm.Pastel1(np.linspace(0, 0.8, n_stats))
    for i in range(n_stats):
        ax.axvspan(i - 0.5, i + 0.5,
                   color=band_colors[i],
                   alpha=0.4,
                   zorder=0)

    for i, m_type in enumerate(MATRIX_TYPES):
        vals = [errors[stat][i] for stat in stats]
        offset = (i - n_types / 2 + 0.5) * bar_width

        ax.bar(x + offset, vals,
               width=bar_width,
               label=m_type,
               color=simple_colors[i])

    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylim(-10, 10)

    ax.set_xticks(x)
    ax.set_xticklabels(stats, rotation=30, ha="right")

    ax.set_ylabel("relative error")
    ax.set_title("Stats estimation relative error")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.legend(title="matrix type", ncol=3)
    plt.tight_layout()

    out = "benchmarks/stats_chart.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

if __name__ == "__main__":
    errors = load_errors()
    if errors is None:
        errors = compute_errors()
        save_errors(errors)
    plot(errors)