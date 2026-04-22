"""
Analyse and visualise qnetwork_results_*.json files from all rollouts_* folders.

Plots produced (saved to DATA_DIR/plots_qnetwork/):
  1. success_rate_per_task.png        – bar chart: success rate per run × task
    1b. avg_success_rate_all_tasks.png  – bar chart: average success rate per run across tasks
    1c. compute_vs_success.png          – scatter: compute (Avg Search Steps) vs average success rate
    1d. compute_vs_success_1d.png       – 1D compute plot with success-rate annotations
  2. episode_length_per_task.png      – bar chart: avg episode length per run × task
  3. q_steps_above_thresh.png         – bar chart: avg steps with Q > thresh per run × task
  4. q_value_hist_per_task.png        – violin/box of final Q-value per run × task
  5. q_value_mean_vs_success.png      – scatter: mean episode Q-value vs success flag (all runs)
"""

import glob
import json
import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    COLOR_PALETTE = sns.color_palette("Set2", n_colors=8)
except ImportError:
    sns = None
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot")
    COLOR_PALETTE = plt.cm.Set2(np.linspace(0, 1, 8))

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get("TDQC_DATA_ROOT", ".")
OUT_DIR   = os.environ.get("TDQC_PLOT_DIR", "./plots_beam_search_expr")
os.makedirs(OUT_DIR, exist_ok=True)

# Short display names for each folder (used by all plots except plot 1c variant)
RUN_LABELS = {
    "rollouts_baseline_old":                     "Baseline",
    # "rollouts_baseline_temp1_5":                "Baseline (T=1.5)",
    "rollouts_grid_n10_temp1.5_probs_ent":      "Probs Entropy",
    "rollouts_grid_n10_temp1.5_BCE":            "RNN top-10 (BCE)",
    "rollouts_grid_n10_temp1.5":                "RNN top-10 (TDQC)",
    "rollouts_grid_n10_temp1.5_thresh0.35":     "RNN top-10 (TDQC, Thresh=0.35)",
}

# Extended labels including all threshold variants — used only by plot_compute_vs_success_thresh_line
THRESH_LINE_RUN_LABELS = {
    **RUN_LABELS,
    "rollouts_grid_n10_temp1.5_thresh0.3":       "RNN top-10 (TDQC, Thresh=0.3)",
    "rollouts_grid_n10_temp1.5_thresh0.35":      "RNN top-10 (TDQC, Thresh=0.35)",
    "rollouts_grid_n10_temp1.5_thresh0.4":       "RNN top-10 (TDQC, Thresh=0.4)",
    "rollouts_grid_n10_temp1.5_thresh0.5":       "RNN top-10 (TDQC, Thresh=0.5)",
    "rollouts_grid_n10_temp1.5_thresh0.6":       "RNN top-10 (TDQC, Thresh=0.6)",
}

# Seed suffixes used to find seed-variant folders for averaging
SEED_SUFFIXES = ["", "_seed8", "_seed9", "_seed10", "_seed11", "_seed12"]  # default suffixes for seed variants (if folder naming follows this pattern)

TDQC_STYLE_RUNS = {
    "rollouts_grid_n10_temp1.5",
    "rollouts_grid_n10_temp1.5_thresh0.35",
}

# Runs that use action-search (all threshold grid runs, excl. baselines/ablations).
# Also defines the ordered threshold series for the compute-vs-success line plot.
_BASELINE_RUNS = {"rollouts_baseline_old", #"rollouts_baseline_temp1_5",
                  "rollouts_grid_n10_temp1.5_probs_ent", "rollouts_grid_n10_temp1.5_BCE"}
AVG_ACTION_SEARCH_RUNS = set(THRESH_LINE_RUN_LABELS.keys()) - _BASELINE_RUNS
THRESH_LINE_RUNS = [r for r in THRESH_LINE_RUN_LABELS if r not in _BASELINE_RUNS]

TASK_SHORT = {
    3: "Task 3\n(bowl→drawer)",
    4: "Task 4\n(mugs→plates)",
    9: "Task 9\n(mug→microwave)",
}

COLORS = ["#2E86AB", "#5E911A", "#820263", "#820263", "#E8630A", "#8E4818", "#C97D4E", "#4E9C81", "#2E86AB", "#94C05B", "#820263", "#555555", "#2F3A44"]
LINE_STYLES = ["--" if run in TDQC_STYLE_RUNS else "-" for run in RUN_LABELS.keys()]
HATCHES     = ["//"  if run in TDQC_STYLE_RUNS else ""  for run in RUN_LABELS.keys()]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_results(data_root: Path):
    """Return list of dicts: {run, label, task_id, task_short, data}"""
    records = []
    pattern = str(data_root / "rollouts_*" / "qnetwork_results_*.json")
    for fpath in sorted(glob.glob(pattern)):
        print(f"Loading: {fpath}")
        run = os.path.basename(os.path.dirname(fpath))
        label = RUN_LABELS.get(run, run)
        with open(fpath) as f:
            tasks = json.load(f)
        for task in tasks:
            records.append({
                "run":        run,
                "label":      label,
                "task_id":    task["task_id"],
                "task_short": TASK_SHORT.get(task["task_id"], f"Task {task['task_id']}"),
                "data":       task,
            })
    return records


# ---------------------------------------------------------------------------
# Helper: grouped bar chart
# ---------------------------------------------------------------------------
def grouped_bar(
    ax,
    run_labels,
    task_ids,
    values,
    yerrs=None,
    ylabel="",
    title="",
    colors=None,
    hatches=None,
    xlabel="",
):
    """
    values[run_idx][task_idx] -> float
    yerrs[run_idx][task_idx]  -> float (optional)
    """
    _colors  = colors  if colors  is not None else COLORS
    _hatches = hatches if hatches is not None else HATCHES
    n_runs  = len(run_labels)
    n_tasks = len(task_ids)
    x       = np.arange(n_tasks)
    width   = 0.8 / n_runs

    for i, (label, color, hatch) in enumerate(zip(run_labels, _colors, _hatches)):
        offsets = x + (i - (n_runs - 1) / 2) * width
        vals    = [values[i][j] for j in range(n_tasks)]
        errs    = [yerrs[i][j]  for j in range(n_tasks)] if yerrs else None
        bars = ax.bar(offsets, vals, width=width * 0.9, label=label.replace("\n", " "),
                      color=color, alpha=0.85, hatch=hatch,
                      yerr=errs, capsize=4, error_kw={"linewidth": 1.2})
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.03,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_SHORT.get(tid, f"Task {tid}") for tid in task_ids],
                       fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=11, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


# ---------------------------------------------------------------------------
# Plot 1: Success rate
# ---------------------------------------------------------------------------
def plot_success_rate(records):
    runs    = list(RUN_LABELS.keys())
    labels  = [RUN_LABELS[r] for r in runs]
    task_ids = sorted({r["task_id"] for r in records})

    values = []
    for run in runs:
        row = []
        for tid in task_ids:
            match = [r for r in records if r["run"] == run and r["task_id"] == tid]
            if match:
                successes = match[0]["data"]["successes"]
                row.append(np.mean(successes))
            else:
                row.append(0.0)
        values.append(row)

    fig, ax = plt.subplots(figsize=(10, 5))
    grouped_bar(ax, labels, task_ids, values,
                ylabel="Success Rate", title="")
    ax.legend(loc="upper left", fontsize=11, ncol=2)

    ax.set_ylim(0, 0.8)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "beam_search_expr_success_rate_per_task.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 1b: Average success rate across tasks (one bar per run)
# ---------------------------------------------------------------------------
def plot_avg_success_rate_all_tasks(records):
    runs = list(RUN_LABELS.keys())
    labels = [RUN_LABELS[r] for r in runs]
    # remove Baseline T = 1.5 and Probs Entropy from this plot since they are not the main focus and have similar performance to the Baseline.
    runs = [r for r in runs if r not in ["rollouts_grid_n10_temp1.5_probs_ent"]]
    labels = [RUN_LABELS[r] for r in runs]
    # also change colors and hatches to match the main plot
    run_order = list(RUN_LABELS.keys())
    run_colors = [COLORS[run_order.index(run)] for run in runs]
    run_hatches = [HATCHES[run_order.index(run)] for run in runs]
    avg_success_rates = []
    avg_action_search_steps = []
    for run in runs:
        run_records = [r for r in records if r["run"] == run]
        per_task_success = []
        all_action_search_steps = []
        for rec in run_records:
            successes = rec["data"]["successes"]
            if successes:
                per_task_success.append(np.mean(successes))

            if run in AVG_ACTION_SEARCH_RUNS:
                all_action_search_steps.extend(rec["data"].get("q_steps_above_thresh", []))

        avg_success_rates.append(float(np.mean(per_task_success)) if per_task_success else 0.0)
        if run in AVG_ACTION_SEARCH_RUNS and all_action_search_steps:
            avg_action_search_steps.append(float(np.mean(all_action_search_steps)))
        else:
            avg_action_search_steps.append(np.nan)

    x = np.arange(len(runs))

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(avg_success_rates) + 2), 6))
    bars = ax.bar(x, avg_success_rates, width=0.6, color=run_colors, alpha=0.85)
    for bar, hatch in zip(bars, run_hatches):
        if hatch:
            bar.set_hatch(hatch)

    label_offset = 0.04
    for bar, val in zip(bars, avg_success_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )


    # Baseline reference line
    ax.axhline(avg_success_rates[0], color="#383862", linestyle="--", linewidth=1.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("\n", " ") for label in labels],
                       rotation=20, ha="right", fontsize=14)
    ax.set_ylabel("Average Success Rate (%)")
    ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.grid(True, linestyle="-", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Overlay average action-search steps only for selected runs.
    x_action = [i for i, run in enumerate(runs)
                if run in AVG_ACTION_SEARCH_RUNS and not np.isnan(avg_action_search_steps[i])]
    if x_action:
        y_action = [ avg_action_search_steps[i] for i in x_action]
        action_color = "#2F3A44"
        ax2 = ax.twinx()
        # Keep secondary axis visible above bars without hiding the primary plot.
        ax2.set_zorder(ax.get_zorder() + 2)
        ax2.patch.set_visible(False)
        ax2.plot(
            x_action,
            y_action,
            color=action_color,
            marker="o",
            linewidth=2.2,
            alpha=0.95,
            label="Compute (Avg Action-Search Steps per Episode)",
            markerfacecolor=action_color,
            markeredgecolor=action_color,
            markeredgewidth=0,
            zorder=7,
        )
        ax2.set_ylabel("Compute (Avg Action-Search Steps per Episode)", color=action_color)
        ax2.tick_params(axis="y", colors=action_color)
        ax2.spines["right"].set_color(action_color)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        action_axis_min = 200
        action_axis_max = 450
        if action_axis_max > action_axis_min:
            ax2.set_ylim(action_axis_min, action_axis_max * 1.15)
        else:
            ax2.set_ylim(0, action_axis_min * 1.1)
        ax2.set_xlim(ax.get_xlim())
        ax2.grid(False)
        ax2.legend(loc="upper right", fontsize=11, framealpha=0.75)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "beam_search_expr_avg_success_rate_all_tasks.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 1c: Compute (Avg Search Steps) vs Avg Success Rate
# ---------------------------------------------------------------------------
def plot_compute_vs_success(records, exclude_runs=None, out_name="beam_search_expr_compute_vs_success.png"):
    runs = list(RUN_LABELS.keys())
    if exclude_runs:
        runs = [r for r in runs if r not in exclude_runs]

    run_order = list(RUN_LABELS.keys())
    run_colors = [COLORS[run_order.index(run)] for run in runs]
    markers = ["o", "s", "^", "D", "P", "*", "*", "*", "*", "*"]

    avg_success_rates = []
    avg_search_steps = []

    for run in runs:
        run_records = [r for r in records if r["run"] == run]

        per_task_success = []
        all_search_steps = []
        all_episode_lengths = []
        for rec in run_records:
            successes = rec["data"].get("successes", [])
            if successes:
                per_task_success.append(np.mean(successes))
            all_episode_lengths.extend(rec["data"].get("episode_lengths", []))
            all_search_steps.extend(rec["data"].get("q_steps_above_thresh", []))

        avg_success = float(np.mean(per_task_success)) if per_task_success else 0.0
        # Keep baseline anchored at zero compute on the x-axis.
        if "baseline" in run:
            avg_steps = 0.0
        elif run in AVG_ACTION_SEARCH_RUNS:
            if all_search_steps:
                avg_steps = float(np.mean(all_search_steps))
            elif all_episode_lengths:
                avg_steps = float(np.mean(all_episode_lengths))
            else:
                avg_steps = 0.0
        else:
            if all_episode_lengths:
                avg_steps = float(np.mean(all_episode_lengths))
            elif all_search_steps:
                avg_steps = float(np.mean(all_search_steps))
            else:
                avg_steps = 0.0

        avg_success_rates.append(avg_success)
        avg_search_steps.append(avg_steps)

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, run in enumerate(runs):
        label = RUN_LABELS[run].replace("\n", " ")
        ax.scatter(
            avg_search_steps[i],
            avg_success_rates[i],
            s=130,
            marker=markers[i % len(markers)],
            color=run_colors[i],
            edgecolors="black",
            linewidths=0.7,
            alpha=0.95,
            label=label,
        )

    ax.set_xlabel("Compute (Avg additional simulation steps)")
    ax.set_ylabel("Average Success Rate")
    ax.set_ylim(0.35, 0.65)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="-", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, out_name)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 1c variant: Compute vs Success with threshold series connected by a line
# ---------------------------------------------------------------------------
def plot_compute_vs_success_thresh_line(records, exclude_runs=None, out_name="beam_search_expr_compute_vs_success_thresh_line.png"):
    all_runs = list(THRESH_LINE_RUN_LABELS.keys())
    if exclude_runs:
        all_runs = [r for r in all_runs if r not in exclude_runs]

    # Marker/color per threshold base run
    THRESH_STYLE = {
        "rollouts_grid_n10_temp1.5":            ("*", "#2E86AB"),
        "rollouts_grid_n10_temp1.5_thresh0.3":  ("*", "#5E911A"),
        "rollouts_grid_n10_temp1.5_thresh0.35": ("*", "#E8630A"),
        "rollouts_grid_n10_temp1.5_thresh0.4":  ("*", "#820263"),
        "rollouts_grid_n10_temp1.5_thresh0.5":  ("*", "#8E4818"),
        "rollouts_grid_n10_temp1.5_thresh0.6":  ("*", "#4E9C81"),
    }

    def _compute_run_stats(run):
        run_records = [r for r in records if r["run"] == run]
        per_task_success, all_search_steps, all_episode_lengths = [], [], []
        for rec in run_records:
            successes = rec["data"].get("successes", [])
            if successes:
                per_task_success.append(np.mean(successes))
            all_episode_lengths.extend(rec["data"].get("episode_lengths", []))
            all_search_steps.extend(rec["data"].get("q_steps_above_thresh", []))
        avg_success = float(np.mean(per_task_success)) if per_task_success else None
        # Check if this run (or its base without seed suffix) is an action-search run
        is_action_search = run in AVG_ACTION_SEARCH_RUNS or any(
            run.startswith(base) for base in AVG_ACTION_SEARCH_RUNS
        )
        if "baseline" in run:
            avg_steps = 0.0
        elif is_action_search:
            avg_steps = float(np.mean(all_search_steps)) if all_search_steps else (float(np.mean(all_episode_lengths)) if all_episode_lengths else 0.0)
        else:
            avg_steps = float(np.mean(all_episode_lengths)) if all_episode_lengths else (float(np.mean(all_search_steps)) if all_search_steps else 0.0)
        return avg_success, avg_steps

    # Build plot points: for each base run, average across seed variants
    # and compute uncertainty as std/sqrt(n_seeds).
    plot_points = []  # list of (label, marker, color, avg_success, avg_steps, success_err, steps_err)
    for run in all_runs:
        suffixes = SEED_SUFFIXES
        seed_runs = [run + sfx for sfx in suffixes]
        seed_stats = [_compute_run_stats(sr) for sr in seed_runs]
        valid = [(s, x) for s, x in seed_stats if s is not None]

        n_valid = len(valid)
        if n_valid > 1:
            parts_s = [s for s, _ in valid]
            parts_x = [x for _, x in valid]
            avg_success = float(np.mean(parts_s))
            avg_steps = float(np.mean(parts_x))
            success_err = float(np.std(parts_s, ddof=1) / np.sqrt(n_valid))
            steps_err = float(np.std(parts_x, ddof=1) / np.sqrt(n_valid))
            print(f"[avg {n_valid} seeds] {run}")
            for sr, (s, x) in zip(seed_runs, seed_stats):
                if s is not None:
                    print(f"  {sr}: success={s:.4f}  compute={x:.2f}")
                else:
                    print(f"  {sr}: no data")
            print(
                f"  => avg success={avg_success:.4f} ± {success_err:.4f}"
                f"  avg compute={avg_steps:.2f} ± {steps_err:.2f}"
            )
        elif valid:
            avg_success, avg_steps = valid[0]
            success_err, steps_err = 0.0, 0.0
        else:
            avg_success, avg_steps = 0.0, 0.0
            success_err, steps_err = 0.0, 0.0

        marker, color = THRESH_STYLE.get(run, ("o", COLORS[all_runs.index(run) % len(COLORS)]))
        label = THRESH_LINE_RUN_LABELS[run].replace("\n", " ")
        plot_points.append((label, marker, color, avg_success, avg_steps, success_err, steps_err))

    runs       = [p[0] for p in plot_points]
    markers_pp = [p[1] for p in plot_points]
    colors_pp  = [p[2] for p in plot_points]
    avg_success_rates = [p[3] for p in plot_points]
    avg_search_steps  = [p[4] for p in plot_points]
    success_errs = [p[5] for p in plot_points]
    steps_errs = [p[6] for p in plot_points]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot averaged points
    for i, label in enumerate(runs):
        ax.scatter(
            avg_search_steps[i],
            avg_success_rates[i],
            s=130,
            marker=markers_pp[i],
            color=colors_pp[i],
            edgecolors="black",
            linewidths=0.7,
            alpha=0.95,
            label=label,
            zorder=3,
        )
        if steps_errs[i] > 0.0 or success_errs[i] > 0.0:
            ax.errorbar(
                avg_search_steps[i],
                avg_success_rates[i],
                xerr=steps_errs[i],
                yerr=success_errs[i],
                fmt="none",
                ecolor=colors_pp[i],
                elinewidth=1.4,
                capsize=3,
                alpha=0.9,
                zorder=2.8,
            )

    # Annotate each point with its success rate
    for i in range(len(runs)):
        ax.annotate(
            f"{avg_success_rates[i]:.2f}",
            (avg_search_steps[i], avg_success_rates[i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color="black",
        )

    # Fit and draw a linear regression line through the threshold series
    thresh_base_runs = list(THRESH_LINE_RUNS)
    thresh_x, thresh_y = [], []
    for base in thresh_base_runs:
        if base not in all_runs:
            continue
        orig_idx = [j for j, pp in enumerate(plot_points) if pp[0] == THRESH_LINE_RUN_LABELS.get(base, base).replace("\n", " ")]
        if orig_idx:
            idx = orig_idx[0]
            thresh_x.append(avg_search_steps[idx])
            thresh_y.append(avg_success_rates[idx])
    thresh_x, thresh_y = np.array(thresh_x, dtype=float), np.array(thresh_y, dtype=float)
    finite = np.isfinite(thresh_x) & np.isfinite(thresh_y)
    thresh_x, thresh_y = thresh_x[finite], thresh_y[finite]
    if len(thresh_x) > 1 and np.ptp(thresh_x) > 0:
        coeffs = np.polyfit(thresh_x, thresh_y, 1)
        x_fit = np.linspace(min(thresh_x), max(thresh_x), 200)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, color="#555555", linewidth=1.5,
                linestyle="-", alpha=0.7, zorder=2)

    ax.set_xlabel("Compute (Avg additional simulation steps)")
    ax.set_ylabel("Average Success Rate")
    ax.set_ylim(0.35, 0.6)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="-", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, out_name)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: Average episode length
# ---------------------------------------------------------------------------
def plot_episode_length(records):
    runs     = list(RUN_LABELS.keys())
    labels   = [RUN_LABELS[r] for r in runs]
    task_ids = sorted({r["task_id"] for r in records})

    values = []
    yerrs  = []
    for run in runs:
        row_mean, row_std = [], []
        for tid in task_ids:
            match = [r for r in records if r["run"] == run and r["task_id"] == tid]
            if match:
                lengths = match[0]["data"]["episode_lengths"]
                row_mean.append(np.mean(lengths))
                row_std.append(np.std(lengths))
            else:
                row_mean.append(0.0)
                row_std.append(0.0)
        values.append(row_mean)
        yerrs.append(row_std)

    fig, ax = plt.subplots(figsize=(10, 5))
    grouped_bar(ax, labels, task_ids, values, yerrs=yerrs,
                ylabel="Avg Episode Length (steps)")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "beam_search_expr_episode_length_per_task.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: Avg Q-steps above threshold
# ---------------------------------------------------------------------------
def plot_q_steps_above_thresh(records):
    all_runs = list(RUN_LABELS.keys())
    runs     = [r for r in all_runs if "baseline" not in r]
    labels   = [RUN_LABELS[r] for r in runs]
    run_colors  = [COLORS[all_runs.index(r)]  for r in runs]
    run_hatches = [HATCHES[all_runs.index(r)] for r in runs]
    task_ids = sorted({r["task_id"] for r in records})

    values = []
    yerrs  = []
    for run in runs:
        row_mean, row_std = [], []
        for tid in task_ids:
            match = [r for r in records if r["run"] == run and r["task_id"] == tid]
            if match:
                steps = match[0]["data"]["q_steps_above_thresh"]
                row_mean.append(np.mean(steps))
                row_std.append(np.std(steps))
            else:
                row_mean.append(0.0)
                row_std.append(0.0)
        values.append(row_mean)
        yerrs.append(row_std)

    fig, ax = plt.subplots(figsize=(10, 5))
    grouped_bar(ax, labels, task_ids, values, yerrs=yerrs,
                ylabel="Avg Additional Simulation Steps per Episode",
                colors=run_colors, hatches=run_hatches)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "beam_search_expr_q_steps_above_thresh.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 5: Distribution of final Q-value per run × task (box / violin)
# ---------------------------------------------------------------------------
def plot_final_q_distribution(records):
    task_ids = sorted({r["task_id"] for r in records})
    runs     = list(RUN_LABELS.keys())
    n_tasks  = len(task_ids)

    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 5), sharey=True)
    if n_tasks == 1:
        axes = [axes]

    for ax, tid in zip(axes, task_ids):
        data_by_run   = []
        label_by_run  = []
        colors_by_run = []

        for run, label, color in zip(runs, RUN_LABELS.values(), COLORS):
            match = [r for r in records if r["run"] == run and r["task_id"] == tid]
            if not match:
                continue
            q_values = match[0]["data"]["q_values"]
            finals   = [ep[-1] for ep in q_values if ep]
            if not finals:
                continue
            data_by_run.append(finals)
            label_by_run.append(label.replace("\n", " "))
            colors_by_run.append(color)

        if not data_by_run:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            parts = ax.violinplot(data_by_run, positions=range(len(data_by_run)),
                                  showmedians=True, showextrema=True)
            for i, (body, color) in enumerate(zip(parts["bodies"], colors_by_run)):
                body.set_facecolor(color)
                body.set_alpha(0.7)
            parts["cmedians"].set_color("black")
            parts["cmaxes"].set_color("gray")
            parts["cmins"].set_color("gray")
            parts["cbars"].set_color("gray")

        ax.set_xticks(range(len(label_by_run)))
        ax.set_xticklabels(label_by_run, rotation=20, ha="right", fontsize=16)
        ax.set_title(TASK_SHORT.get(tid, f"Task {tid}"))
        ax.set_ylim(0, 1)
        if ax is axes[0]:
            ax.set_ylabel("Final Q-value")

    fig.suptitle("Distribution of Final Q-value per Episode", fontsize=15, y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "beam_search_expr_q_value_final_dist_per_task.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 7: Mean episode Q-value vs success (scatter, all runs pooled)
# ---------------------------------------------------------------------------
def plot_mean_q_vs_success(records):
    fig, axes = plt.subplots(1, len(RUN_LABELS), figsize=(5 * len(RUN_LABELS), 4),
                             sharey=True, sharex=True)

    for ax, (run, label) in zip(axes, RUN_LABELS.items()):
        run_records = [r for r in records if r["run"] == run]
        mean_qs, successes, colors_ep = [], [], []

        for rec in run_records:
            q_values = rec["data"]["q_values"]
            succ     = rec["data"]["successes"]
            tid      = rec["task_id"]
            color    = COLORS[list(RUN_LABELS.keys()).index(run)]

            for ep_q, s in zip(q_values, succ):
                if ep_q:
                    mean_qs.append(np.mean(ep_q))
                    successes.append(int(s))
                    colors_ep.append(tid)

        mean_qs   = np.array(mean_qs)
        successes = np.array(successes)
        task_arr  = np.array(colors_ep)

        for tid, marker, tshort in [(3, "o", "T3"), (4, "s", "T4"), (9, "^", "T9")]:
            mask   = task_arr == tid
            c_succ = np.where(successes[mask], "#2ca02c", "#d62728")
            ax.scatter(mean_qs[mask], successes[mask] + np.random.uniform(-0.05, 0.05, mask.sum()),
                       c=c_succ, marker=marker, alpha=0.6, s=40, label=tshort)

        ax.set_xlabel("Mean Episode Q-value")
        ax.set_title(label.replace("\n", " "), fontsize=16)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Failure", "Success"])
        ax.legend(fontsize=9)

    fig.suptitle("Mean Episode Q-value vs Outcome (green=success, red=failure)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "beam_search_expr_q_value_mean_vs_success.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global OUT_DIR

    parser = argparse.ArgumentParser(description="Plot guided action search results")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Path containing rollout_* folders",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for saved figures",
    )
    args = parser.parse_args()

    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading results …")
    records = load_all_results(args.data_root)
    print(f"Loaded {len(records)} task-run combinations")

    plot_success_rate(records)
    plot_compute_vs_success(records, exclude_runs=["rollouts_grid_n10_temp1.5_probs_ent"],
                            out_name="beam_search_expr_compute_vs_success.png")
    plot_compute_vs_success(records, out_name="beam_search_expr_compute_vs_success_all_tasks.png")
    plot_compute_vs_success_thresh_line(records, # exclude_runs=["rollouts_baseline_temp1_5", "rollouts_grid_n10_temp1.5_probs_ent"],
                                        out_name="beam_search_expr_compute_vs_success_thresh_line.png")
    plot_episode_length(records)
    plot_q_steps_above_thresh(records)
    plot_final_q_distribution(records)
    plot_mean_q_vs_success(records)

    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
