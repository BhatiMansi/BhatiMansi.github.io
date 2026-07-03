#!/usr/bin/env python3
"""Generate blog-oriented Phase 4 split-1 figures.

This script compares three PS-solve modes:

* old no-RBM baseline, subspace=300
* fresh no-RBM control, subspace=100
* RBM-polished run, subspace=100

It intentionally reads the already produced performance CSV for the
old-baseline/RBM pair and parses the fresh SLURM output for the subspace-100
control, so the figures can be regenerated without re-running the solver.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-phase4-blog")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from S_3D.analyze_phase4_split_performance import parse_log, region_for


REGION_ORDER = [
    "center seed",
    "first lower",
    "second lower",
    "mature lower",
    "first upper",
    "second upper",
    "mature upper",
]

COLORS = {
    "baseline300": "#3b5b92",
    "norun100": "#c4562e",
    "rbm": "#22936f",
    "accent": "#7c3f98",
    "ink": "#222222",
    "muted": "#71717a",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfd",
            "axes.edgecolor": "#d4d4d8",
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "axes.grid": True,
            "grid.color": "#e4e4e7",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.9,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def save_figure(fig: plt.Figure, outdir: Path, stem: str, made: list[str]) -> None:
    png = outdir / f"{stem}.png"
    svg = outdir / f"{stem}.svg"
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    plt.close(fig)
    made.extend([png.name, svg.name])


def read_metrics(path: Path) -> list[dict[str, object]]:
    numeric = {
        "R",
        "P",
        "rbm_cycles",
        "baseline_cycles",
        "cycle_speedup",
        "cycle_saved",
        "rbm_time_s",
        "baseline_time_s",
        "time_speedup",
        "time_saved_s",
        "max_abs_energy_diff",
    }
    rows: list[dict[str, object]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, object] = {}
            for key, value in row.items():
                if key in numeric:
                    parsed[key] = float(value)
                    if key in {"R", "P", "rbm_cycles", "baseline_cycles", "cycle_saved"}:
                        parsed[key] = int(float(value))
                else:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def ps_records_by_key(log_path: Path) -> tuple[dict[tuple[int, int], dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    _, records, _, block_times = parse_log(log_path)
    ps_records = [r for r in records if r["phase"] == "ps"]
    return {(int(r["R"]), int(r["P"])): r for r in ps_records}, ps_records, block_times


def read_residuals(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    out: dict[tuple[int, int], dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["R"]), int(row["P"]))
            out[key] = {
                "max": float(row["max_rbm_residual"]),
                "mean": float(row["mean_rbm_residual"]),
            }
    return out


def merge_rows(metrics: list[dict[str, object]], no_rbm100: dict[tuple[int, int], dict[str, object]], rbm: dict[tuple[int, int], dict[str, object]]) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for row in metrics:
        key = (int(row["R"]), int(row["P"]))
        if key not in no_rbm100:
            continue
        out = dict(row)
        out["no_rbm100_cycles"] = int(no_rbm100[key]["cycles"])
        out["no_rbm100_time_s"] = float(no_rbm100[key]["time_s"])
        if key in rbm:
            energies_new = np.asarray(no_rbm100[key]["energies"], dtype=float)
            energies_rbm = np.asarray(rbm[key]["energies"], dtype=float)
            out["new_vs_rbm_energy_diff"] = float(np.nanmax(np.abs(energies_new - energies_rbm)))
        else:
            out["new_vs_rbm_energy_diff"] = math.nan
        out["no_rbm100_vs_rbm_speedup"] = float(out["no_rbm100_time_s"]) / float(out["rbm_time_s"])
        out["baseline300_vs_no_rbm100_speedup"] = float(out["baseline_time_s"]) / float(out["no_rbm100_time_s"])
        merged.append(out)
    return merged


def to_grid(rows: list[dict[str, object]], field: str) -> np.ndarray:
    max_r = max(int(r["R"]) for r in rows)
    max_p = max(int(r["P"]) for r in rows)
    grid = np.full((max_r + 1, max_p + 1), np.nan)
    for row in rows:
        grid[int(row["R"]), int(row["P"])] = float(row[field])
    return grid


def totals_by_r(rows: list[dict[str, object]], field: str) -> tuple[np.ndarray, np.ndarray]:
    grouped: defaultdict[int, float] = defaultdict(float)
    for row in rows:
        grouped[int(row["R"])] += float(row[field])
    keys = np.array(sorted(grouped), dtype=int)
    vals = np.array([grouped[int(k)] for k in keys], dtype=float)
    return keys, vals


def region_medians(rows: list[dict[str, object]], field: str) -> list[float]:
    vals: list[float] = []
    for region in REGION_ORDER:
        subset = [float(r[field]) for r in rows if r["region"] == region]
        vals.append(float(np.median(subset)) if subset else math.nan)
    return vals


def plot_cumulative_time(rows: list[dict[str, object]], outdir: Path, made: list[str]) -> None:
    x = np.arange(1, len(rows) + 1)
    series = [
        ("No RBM, subspace 300", "baseline_time_s", COLORS["baseline300"]),
        ("No RBM, subspace 100", "no_rbm100_time_s", COLORS["norun100"]),
        ("RBM-polished, subspace 100", "rbm_time_s", COLORS["rbm"]),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for label, field, color in series:
        y = np.cumsum([float(r[field]) for r in rows]) / 3600.0
        ax.plot(x, y, lw=2.4, color=color, label=f"{label}: {y[-1]:.2f} h")
        ax.scatter([x[-1]], [y[-1]], s=28, color=color, zorder=3)
    ax.set_title("Cumulative Davidson Time Across the Split")
    ax.set_xlabel("R/P grid point, sorted by (R, P)")
    ax.set_ylabel("Cumulative PS Davidson time (hours)")
    ax.legend(loc="upper left")
    ax.margins(x=0.01)
    save_figure(fig, outdir, "triple_cumulative_davidson_time", made)


def plot_cumulative_savings(rows: list[dict[str, object]], outdir: Path, made: list[str]) -> None:
    x = np.arange(1, len(rows) + 1)
    old_minus_rbm = np.cumsum([float(r["baseline_time_s"]) - float(r["rbm_time_s"]) for r in rows]) / 3600.0
    new_minus_rbm = np.cumsum([float(r["no_rbm100_time_s"]) - float(r["rbm_time_s"]) for r in rows]) / 3600.0
    old_minus_new = np.cumsum([float(r["baseline_time_s"]) - float(r["no_rbm100_time_s"]) for r in rows]) / 3600.0

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.fill_between(x, old_minus_rbm, color=COLORS["rbm"], alpha=0.14)
    ax.plot(x, old_minus_rbm, lw=2.5, color=COLORS["rbm"], label=f"old baseline -> RBM: {old_minus_rbm[-1]:.2f} h")
    ax.plot(x, new_minus_rbm, lw=2.3, color=COLORS["accent"], label=f"subspace 100 control -> RBM: {new_minus_rbm[-1]:.2f} h")
    ax.plot(x, old_minus_new, lw=2.0, color=COLORS["norun100"], label=f"old baseline -> subspace 100: {old_minus_new[-1]:.2f} h")
    ax.set_title("Cumulative Time Saved")
    ax.set_xlabel("R/P grid point, sorted by (R, P)")
    ax.set_ylabel("Cumulative PS Davidson time saved (hours)")
    ax.legend(loc="upper left")
    ax.margins(x=0.01)
    save_figure(fig, outdir, "cumulative_time_saved_three_way", made)


def plot_per_r_totals(rows: list[dict[str, object]], outdir: Path, made: list[str]) -> None:
    r, base = totals_by_r(rows, "baseline_time_s")
    _, no100 = totals_by_r(rows, "no_rbm100_time_s")
    _, rbm = totals_by_r(rows, "rbm_time_s")
    width = 0.26

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(r - width, base / 60.0, width=width, color=COLORS["baseline300"], label="No RBM, subspace 300")
    ax.bar(r, no100 / 60.0, width=width, color=COLORS["norun100"], label="No RBM, subspace 100")
    ax.bar(r + width, rbm / 60.0, width=width, color=COLORS["rbm"], label="RBM-polished")
    ax.set_title("Per-R Davidson Work")
    ax.set_xlabel("R index in split 1")
    ax.set_ylabel("Total PS Davidson time per R row (minutes)")
    ax.set_xticks(r)
    ax.legend(loc="upper left", ncols=3)
    save_figure(fig, outdir, "per_R_total_davidson_time", made)


def plot_region_speedups(rows: list[dict[str, object]], outdir: Path, made: list[str]) -> None:
    y = np.arange(len(REGION_ORDER))
    old_to_rbm = region_medians(rows, "time_speedup")
    new_to_rbm = region_medians(rows, "no_rbm100_vs_rbm_speedup")
    old_to_new = region_medians(rows, "baseline300_vs_no_rbm100_speedup")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.axvline(1.0, color="#a1a1aa", lw=1.0)
    for idx in y:
        ax.plot([old_to_new[idx], old_to_rbm[idx]], [idx, idx], color="#d4d4d8", lw=3, zorder=1)
    ax.scatter(old_to_rbm, y, s=58, color=COLORS["rbm"], label="old baseline / RBM", zorder=3)
    ax.scatter(new_to_rbm, y, s=58, color=COLORS["accent"], label="subspace 100 control / RBM", zorder=3)
    ax.scatter(old_to_new, y, s=44, color=COLORS["norun100"], label="old baseline / subspace 100", zorder=3)
    ax.set_yticks(y, REGION_ORDER)
    ax.set_xscale("log")
    ax.set_xlabel("Median wall-time speedup (log scale)")
    ax.set_title("Where the Speedup Actually Comes From")
    ax.legend(loc="lower right")
    save_figure(fig, outdir, "region_speedup_lollipop", made)


def plot_speedup_heatmap(rows: list[dict[str, object]], outdir: Path, made: list[str]) -> None:
    grid = to_grid(rows, "no_rbm100_vs_rbm_speedup")
    fig, ax = plt.subplots(figsize=(8.5, 4.1))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="magma", vmin=1.0, vmax=float(np.nanpercentile(grid, 99)))
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("subspace 100 control / RBM time")
    ax.set_title("RBM Payoff Map Over the R/P Grid")
    ax.set_xlabel("P index")
    ax.set_ylabel("R index")
    save_figure(fig, outdir, "speedup_heatmap_subspace100_to_rbm", made)


def plot_cycle_heatmaps(rows: list[dict[str, object]], outdir: Path, made: list[str]) -> None:
    fields = [
        ("No RBM, subspace 300", "baseline_cycles"),
        ("No RBM, subspace 100", "no_rbm100_cycles"),
        ("RBM-polished", "rbm_cycles"),
    ]
    vmax = max(float(np.nanpercentile(to_grid(rows, field), 99.5)) for _, field in fields)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), sharex=True, sharey=True)
    im = None
    for ax, (title, field) in zip(axes, fields):
        im = ax.imshow(to_grid(rows, field), origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("P index")
    axes[0].set_ylabel("R index")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, pad=0.02)
        cbar.set_label("Davidson cycles")
    fig.suptitle("Cycle Count Texture Before and After RBM Warm Starts", y=1.02)
    save_figure(fig, outdir, "cycle_heatmaps_three_way", made)


def plot_residual_payoff(rows: list[dict[str, object]], residuals: dict[tuple[int, int], dict[str, float]], outdir: Path, made: list[str]) -> None:
    region_colors = {
        "center seed": "#3b5b92",
        "first lower": "#c4562e",
        "second lower": "#d18f28",
        "mature lower": "#22936f",
        "first upper": "#7c3f98",
        "second upper": "#a8557d",
        "mature upper": "#0f766e",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    for region in REGION_ORDER:
        xs = []
        ys = []
        for row in rows:
            key = (int(row["R"]), int(row["P"]))
            if row["region"] == region and key in residuals:
                xs.append(max(residuals[key]["max"], 1.0e-12))
                ys.append(float(row["no_rbm100_vs_rbm_speedup"]))
        if xs:
            ax.scatter(xs, ys, s=26, alpha=0.78, color=region_colors[region], label=region)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("RBM guess max residual before polish")
    ax.set_ylabel("subspace 100 control / RBM wall-time speedup")
    ax.set_title("Residual Quality vs Runtime Payoff")
    ax.legend(loc="upper left", ncols=2, fontsize=8)
    save_figure(fig, outdir, "rbm_residual_vs_speedup", made)


def plot_energy_agreement(rows: list[dict[str, object]], outdir: Path, made: list[str]) -> None:
    grid = to_grid(rows, "new_vs_rbm_energy_diff")
    grid = np.where(np.isfinite(grid), np.maximum(grid, 1.0e-16), np.nan)
    fig, ax = plt.subplots(figsize=(8.5, 4.1))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="cividis", norm=LogNorm(vmin=1.0e-16, vmax=max(1.0e-12, float(np.nanmax(grid)))))
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("max |E_noRBM100 - E_RBM| from logs")
    ax.set_title("Energy Agreement: Fresh Control vs RBM Run")
    ax.set_xlabel("P index")
    ax.set_ylabel("R index")
    save_figure(fig, outdir, "energy_agreement_heatmap", made)


def scratch_file(scratch_dir: Path, name: str) -> Path:
    prefix = "matrix_spin_110_erf_coulomb_J_0.5_Pth_0.7071_Pph_0.5_a_8000.0_m_2000.0"
    return scratch_dir / f"{prefix}_{name}_split_1.npy"


def load_array(scratch_dir: Path, name: str) -> np.ndarray:
    return np.load(scratch_file(scratch_dir, name))


def finite_populated_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(arrays[0], dtype=bool)
    for arr in arrays:
        mask |= np.abs(arr) > 0
    return mask


def plot_energy_surfaces(scratch_dir: Path, outdir: Path, made: list[str]) -> bool:
    try:
        eps_g = load_array(scratch_dir, "EPSg")
        eps_e = load_array(scratch_dir, "EPSe")
    except OSError:
        return False

    mask = finite_populated_mask(eps_g, eps_e)
    g = np.ma.masked_where(~mask, eps_g)
    e = np.ma.masked_where(~mask, eps_e)
    vmin = float(min(np.nanmin(eps_g[mask]), np.nanmin(eps_e[mask])))
    vmax = float(max(np.nanmax(eps_g[mask]), np.nanmax(eps_e[mask])))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharex=True, sharey=True)
    im = axes[0].imshow(g, origin="lower", aspect="auto", cmap="plasma", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground-state PS energy")
    axes[1].imshow(e, origin="lower", aspect="auto", cmap="plasma", vmin=vmin, vmax=vmax)
    axes[1].set_title("Excited-state PS energy")
    for ax in axes:
        ax.set_xlabel("P index")
    axes[0].set_ylabel("R index")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, pad=0.02)
    cbar.set_label("Energy")
    fig.suptitle("Populated Energy Surface for Split 1", y=1.02)
    save_figure(fig, outdir, "energy_surfaces_gs_es", made)
    return True


def plot_spin_consistency(scratch_dir: Path, outdir: Path, made: list[str]) -> bool:
    try:
        sx_g = load_array(scratch_dir, "exp_sx_gs")
        sy_g = load_array(scratch_dir, "exp_sy_gs")
        sz_g = load_array(scratch_dir, "exp_sz_gs")
        sx_e = load_array(scratch_dir, "exp_sx_es")
        sy_e = load_array(scratch_dir, "exp_sy_es")
        sz_e = load_array(scratch_dir, "exp_sz_es")
    except OSError:
        return False

    spin2_g = sx_g * sx_g + sy_g * sy_g + sz_g * sz_g
    spin2_e = sx_e * sx_e + sy_e * sy_e + sz_e * sz_e
    mask = finite_populated_mask(spin2_g, spin2_e)
    dev_g = np.ma.masked_where(~mask, np.abs(spin2_g - 0.25))
    dev_e = np.ma.masked_where(~mask, np.abs(spin2_e - 0.25))
    vmax = max(1.0e-5, float(max(np.nanmax(dev_g), np.nanmax(dev_e))))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharex=True, sharey=True)
    im = axes[0].imshow(dev_g, origin="lower", aspect="auto", cmap="inferno", norm=LogNorm(vmin=1.0e-8, vmax=vmax))
    axes[0].set_title("Ground-state |<S>^2 - 1/4|")
    axes[1].imshow(dev_e, origin="lower", aspect="auto", cmap="inferno", norm=LogNorm(vmin=1.0e-8, vmax=vmax))
    axes[1].set_title("Excited-state |<S>^2 - 1/4|")
    for ax in axes:
        ax.set_xlabel("P index")
    axes[0].set_ylabel("R index")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, pad=0.02)
    cbar.set_label("absolute deviation")
    fig.suptitle("Spin-Length Consistency Across the Saved Grid", y=1.02)
    save_figure(fig, outdir, "spin_length_consistency", made)
    return True


def block_time(blocks: list[dict[str, object]], name: str) -> float | None:
    vals = [float(b["time_s"]) for b in blocks if b["name"] == name]
    return vals[-1] if vals else None


def write_readme(outdir: Path, made: list[str], rows: list[dict[str, object]], new_blocks: list[dict[str, object]], skipped: list[str]) -> None:
    rbm_total = sum(float(r["rbm_time_s"]) for r in rows)
    old_total = sum(float(r["baseline_time_s"]) for r in rows)
    new_total = sum(float(r["no_rbm100_time_s"]) for r in rows)
    max_energy = max(float(r["new_vs_rbm_energy_diff"]) for r in rows if math.isfinite(float(r["new_vs_rbm_energy_diff"])))
    r_loop = block_time(new_blocks, "R for loop")

    lines = [
        "# Phase 4 Split 1 Blog Figures",
        "",
        "Generated by `make_blog_figures.py`.",
        "",
        "## Quick Numbers",
        "",
        f"- PS solves compared: `{len(rows)}`",
        f"- Old no-RBM subspace-300 PS time: `{old_total:.3f} s` (`{old_total / 3600.0:.2f} h`)",
        f"- Fresh no-RBM subspace-100 PS time: `{new_total:.3f} s` (`{new_total / 3600.0:.2f} h`)",
        f"- RBM-polished PS time: `{rbm_total:.3f} s` (`{rbm_total / 3600.0:.2f} h`)",
        f"- Fresh no-RBM subspace-100 / RBM PS speedup: `{new_total / rbm_total:.3f}x`",
        f"- Old baseline / fresh no-RBM subspace-100 PS speedup: `{old_total / new_total:.3f}x`",
        f"- Max printed-log energy difference, fresh control vs RBM: `{max_energy:.3e}`",
    ]
    if r_loop is not None:
        lines.append(f"- Fresh no-RBM subspace-100 `R for loop`: `{r_loop:.3f} s`")
    if skipped:
        lines.extend(["", "## Skipped", ""])
        lines.extend(f"- {item}" for item in skipped)
    lines.extend(["", "## Artifacts", ""])
    lines.extend(f"- `{name}`" for name in made)
    (outdir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="S_3D/phase4_logs/performance_split1/ps_solve_metrics.csv")
    parser.add_argument("--residuals", default="S_3D/phase4_logs/performance_split1/rbm_residuals.csv")
    parser.add_argument("--rbm-log", default="S_3D/phase4_logs/rbm_production_split1.log")
    parser.add_argument("--no-rbm100-log", default="/home/mb3835/group_storage/slurm-9971703.out")
    parser.add_argument("--scratch-dir", default="/scratch/gpfs/SUBOTNIK/mb3835")
    parser.add_argument("--outdir", default="S_3D/phase4_logs/performance_split1/blog_figures")
    args = parser.parse_args()

    configure_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = read_metrics(Path(args.metrics))
    no_rbm100, _, new_blocks = ps_records_by_key(Path(args.no_rbm100_log))
    rbm, _, _ = ps_records_by_key(Path(args.rbm_log))
    rows = merge_rows(metrics, no_rbm100, rbm)
    residuals = read_residuals(Path(args.residuals))

    if len(rows) != len(metrics):
        raise RuntimeError(f"Only merged {len(rows)} of {len(metrics)} metric rows")

    made: list[str] = []
    skipped: list[str] = []

    plot_cumulative_time(rows, outdir, made)
    plot_cumulative_savings(rows, outdir, made)
    plot_per_r_totals(rows, outdir, made)
    plot_region_speedups(rows, outdir, made)
    plot_speedup_heatmap(rows, outdir, made)
    plot_cycle_heatmaps(rows, outdir, made)
    plot_residual_payoff(rows, residuals, outdir, made)
    plot_energy_agreement(rows, outdir, made)

    scratch_dir = Path(args.scratch_dir)
    if not plot_energy_surfaces(scratch_dir, outdir, made):
        skipped.append("energy surfaces; saved EPSg/EPSe arrays were not readable")
    if not plot_spin_consistency(scratch_dir, outdir, made):
        skipped.append("spin consistency; saved spin expectation arrays were not readable")

    write_readme(outdir, made, rows, new_blocks, skipped)
    print(f"Wrote {len(made)} figure files to {outdir}")
    for name in made:
        print(name)


if __name__ == "__main__":
    main()
