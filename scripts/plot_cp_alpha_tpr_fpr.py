"""Plot split-CP TPR/FPR vs alpha for all mapped methods per experiment.

This script reads rows produced by eval_split_conformal(...), i.e. entries from:

classification_logs.append({
    "detect_method": method_name,
    "cal split": f"{'+'.join(calib_split_names)}",
    "test split": f"{'+'.join(test_split_names)}",
    "calib on": calib_label,
    "task": "all",
    "thresh_method": f"split CP, cal on {'+'.join(calib_split_names)}",
    "alpha": alpha,
    "time": eval_time,
    **result,
    "threshold": thresh_pos,
})

Rows are loaded from W&B tables such as classify_cp_maxsofar/<detect_method>.

Example:
    python scripts/plot_cp_alpha_tpr_fpr.py \
        --project tdqc-final \
        --eval_time "by earliest stop" \
        --calib_on neg \
        --output_dir other_plots/cp_alpha_curves
"""

import argparse
import json
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import wandb

from failure_prob.utils.wandb import load_summary_tables_from_runs


LABEL_MAPPING = {
    # LIBERO methods
    "openvla-10-indep-mlp_BCE": "MLP on features (BCE Loss)",
    "openvla-10-indep-mlp_TD0": "MLP on features (TDQC Loss)",
    "openvla-10-lstm-lstm": "RNN on features (BCE Loss)",
    "openvla-10-lstm-lstm_TD0": "RNN on features (TDQC Loss)",
    "openvla-10-q_learning-top_k_probs_BCE": "RNN on top-10 probabilities (BCE Loss)",
    "openvla-10-q_learning-top_k_probs_TD0": "RNN on top-10 probabilities (TDQC Loss)",

    # WidowX methods
    "openvla-widowx-indep-mlp_BCE": "MLP on features (BCE Loss)",
    "openvla-widowx-indep-mlp_TD0": "MLP on features (TDQC Loss)",
    "openvla-widowx-lstm-lstm": "RNN on features (BCE Loss)",
    "openvla-widowx-lstm-lstm_TD0": "RNN on features (TDQC Loss)",
    "openvla-widowx-q_learning-top_k_probs_BCE": "RNN on top-10 probabilities (BCE Loss)",
    "openvla-widowx-q_learning-top_k_probs": "RNN on top-10 probabilities (TDQC Loss)",

    # Droid methods
    "pizero_fast_droid-0510-indep-mlp_BCE": "MLP on features (BCE Loss)",
    "pizero_fast_droid-0510-indep-mlp_TD0": "MLP on features (TDQC Loss)",
    "pizero_fast_droid-0510-lstm-lstm": "RNN on features (BCE Loss)",
    "pizero_fast_droid-0510-lstm-lstm_TD0": "RNN on features (TDQC Loss)",
    "pizero_fast_droid-0510-lstm-lstm_top_k_probs_BCE": "RNN on top-10 probabilities (BCE Loss)",
    "pizero_fast_droid-0510-lstm-lstm_top_k_probs_TD0": "RNN on top-10 probabilities (TDQC Loss)",

    # Pi0 FAST LIBERO methods
    "pizero_fast-default-indep-mlp_BCE": "MLP on features (BCE Loss)",
    "pizero_fast-default-indep-mlp_TD0": "MLP on features (TDQC Loss)",
    "pizero_fast-default-lstm-lstm": "RNN on features (BCE Loss)",
    "pizero_fast-default-lstm-lstm_TD0": "RNN on features (TDQC Loss)",
    "pizero_fast-default-lstm-lstm_top_k_probs_BCE": "RNN on top-10 probabilities (BCE Loss)",
    "pizero_fast-default-lstm-lstm_top_k_probs_TD0": "RNN on top-10 probabilities (TDQC Loss)",

    # Pi0 LIBERO methods
    "pizero-default-indep-mlp_BCE": "MLP on features (BCE Loss)",
    "pizero-default-indep-mlp_TD0": "MLP on features (TDQC Loss)",
    "pizero-default-lstm-lstm": "RNN on features (BCE Loss)",
    "pizero-default-lstm-lstm_TD0": "RNN on features (TDQC Loss)",

}


COLOR_GROUPS = {
    "mlp":        "#2E86AB",
    "lstm":       "#94C05B",
    "top_k_probs": "#820263",
}


def method_label_to_style(label: str):
    """Return (color, linestyle) from a method label string."""
    if "top-10" in label:
        color = COLOR_GROUPS["top_k_probs"]
    elif "MLP" in label:
        color = COLOR_GROUPS["mlp"]
    else:
        color = COLOR_GROUPS["lstm"]
    linestyle = "--" if "TDQC" in label else "-"
    return color, linestyle


EXPERIMENT_DISPLAY_NAMES = {
    "openvla-10":           "openVLA LIBERO-10",
    "openvla-widowx":       "openVLA WidowX",
    "pizero-default":       "$\pi_0$ LIBERO",
    "pizero_fast-default":  "$\pi_0$-FAST LIBERO",
    "pizero_fast_droid-0510": "$\pi_0$-FAST Droid",
}


def infer_experiment_key(method_key: str) -> str:
    """Infer experiment prefix from a LABEL_MAPPING method key."""
    m = re.match(r"^(.*?)-(indep|lstm|q_learning)-", method_key)
    if m:
        return m.group(1)
    return method_key


_ALLOWED_SUFFIX_RE = re.compile(r"^(_test)?(-\d+)?$")

def find_runs_for_method_key(all_runs: List[wandb.apis.public.Run], method_key: str) -> List[wandb.apis.public.Run]:
    """Match all runs belonging to one method key.

    We accept exact names and common suffix variants like:
    - <method_key>
    - <method_key>-<seed>
    - <method_key>_test
    - <method_key>_test-<seed>

    We explicitly reject names where the suffix looks like another method
    variant (e.g. _TD0), to avoid openvla-10-lstm-lstm matching
    openvla-10-lstm-lstm_TD0 runs.
    """
    matched = []
    for run in all_runs:
        name = run.name
        if not name.startswith(method_key):
            continue
        suffix = name[len(method_key):]
        if _ALLOWED_SUFFIX_RE.match(suffix):
            matched.append(run)

    return matched


def extract_cp_rows(
    run: wandb.apis.public.Run,
    table_prefix: str,
    eval_time: str,
    calib_on: str,
) -> pd.DataFrame:
    """Extract split-CP rows (alpha,tpr,fpr,...) from a run's W&B tables."""
    summary_tables = load_summary_tables_from_runs([run])[0]

    eval_time = "by final end" 
    rows = []
    for key, table_df in summary_tables.items():
        if not key.startswith(f"{table_prefix}/"):
            continue

        if isinstance(table_df, dict) and table_df.get("_type") == "table-file":
            try:
                with run.file(table_df["path"]).download(replace=True) as f:
                    raw = json.load(f)
                table_df = pd.DataFrame(raw["data"], columns=raw["columns"])
            except Exception as e:
                print(f"Failed to download table for key '{key}': {e}")
                continue

        detect_method = key.split("/", 1)[1]
        df = table_df.copy()

        required_cols = {"alpha", "tpr", "fpr"}
        if not required_cols.issubset(df.columns):
            continue

        if "time" in df.columns:
            df = df[df["time"] == eval_time]
        if "calib on" in df.columns:
            df = df[df["calib on"] == calib_on]
        if "thresh_method" in df.columns:
            df = df[df["thresh_method"].astype(str).str.contains("split CP", na=False)]
        df = df[df["test split"] == 'val_unseen']

        if len(df) == 0:
            continue

        df = df.copy()
        df["run_name"] = run.name
        df["run_id"] = run.id
        df["table_key"] = key
        df["detect_method"] = detect_method
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def plot_metric_vs_alpha(
    df_plot: pd.DataFrame,
    metric_key: str,
    output_path: str,
    experiment_title: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=300)

    method_names = sorted(df_plot["method_label"].unique())

    for method_name in method_names:
        df_m = df_plot[df_plot["method_label"] == method_name].sort_values("alpha")
        x = df_m["alpha"].to_numpy()
        y = df_m[f"{metric_key}_mean"].to_numpy()
        yerr = df_m[f"{metric_key}_std"].fillna(0.0).to_numpy()
        color, linestyle = method_label_to_style(method_name)
        ax.plot(x, y, marker="o", linewidth=1.5, markersize=3.5, color=color,
                linestyle=linestyle, label=method_name)
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.12)

    if metric_key == "tnr":
        x_ref = np.linspace(0.0, 1.0, 200)
        ax.plot(x_ref, 1 - x_ref, color="black", linewidth=1.2, linestyle="--",
                label="1 − α (theoretical)", zorder=2)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Significance Level alpha")
    ax.set_ylabel(metric_key.upper())
    title = f"{experiment_title} — {metric_key.upper()} vs α" if experiment_title else f"{metric_key.upper()} vs α"
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    
    handles += [
        mlines.Line2D([0], [0], color="gray", linewidth=1.5, linestyle="-",  label="— BCE"),
        mlines.Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", label="-- TDQC"),
    ]
    ax.legend(handles=handles, frameon=False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


METRICS = ["tpr", "fpr"]
title_map = {
    "TPR": "TPR",
    "FPR": "FPR",
}


def plot_roc_curves(
    data: list,
    method_name: str
):
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {method_name}")

    for fpr, tpr, name, roc_auc in data:
        plt.plot(fpr, tpr, label=f"{name}, AUC={roc_auc*100:.2f}")

    plt.legend()
    return fig


def plot_combined_grid(df_stats: pd.DataFrame, output_path: str) -> None:
    """One figure: rows = experiments, cols = metrics (tpr, fpr, tnr)."""
    experiments = [k for k in EXPERIMENT_DISPLAY_NAMES if k in df_stats["experiment_key"].unique()]
    # also include any experiment keys not in EXPERIMENT_DISPLAY_NAMES
    extra = [e for e in sorted(df_stats["experiment_key"].unique()) if e not in EXPERIMENT_DISPLAY_NAMES]
    experiments = experiments + extra

    n_rows = len(experiments)
    n_cols = len(METRICS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.6 * n_rows),
                             sharex=False, squeeze=False)

    for row_idx, exp_key in enumerate(experiments):
        df_exp = df_stats[df_stats["experiment_key"] == exp_key]
        exp_title = EXPERIMENT_DISPLAY_NAMES.get(exp_key, exp_key)
        method_names = sorted(df_exp["method_label"].unique())

        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx][col_idx]

            for method_name in method_names:
                df_m = df_exp[df_exp["method_label"] == method_name].sort_values("alpha")
                x = df_m["alpha"].to_numpy()
                mean_col = f"{metric}_mean"
                std_col = f"{metric}_std"
                if mean_col not in df_m.columns:
                    continue
                y = df_m[mean_col].to_numpy()
                yerr = df_m[std_col].fillna(0.0).to_numpy()
                color, linestyle = method_label_to_style(method_name)
                ax.plot(x, y, marker="o", linewidth=1.5, markersize=3.5, color=color,
                        linestyle=linestyle, label=method_name)
                ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.12)
                
                if metric == "fpr":
                    x_ref = np.linspace(0.0, 1.0, 200)
                    ax.plot(x_ref, x_ref, color="black", linewidth=1.2, linestyle="--",
                            label="α", zorder=2)
                    
            ax.set_xlim(0.0, 1.0)

            ax.grid(alpha=0.25, linewidth=0.5)
            if col_idx == 0:
                ax.set_ylabel(exp_title)
            if row_idx == 0:
                ax.set_title(title_map[metric.upper()])
            if row_idx == n_rows - 1:
                ax.set_xlabel("α")
    # Collect legend handles from first experiment/col that has data
    legend_handles = None
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            h, _ = axes[row_idx][col_idx].get_legend_handles_labels()
            if h:
                legend_handles = h
                break
        if legend_handles:
            break
    if legend_handles is None:
        legend_handles = []
    legend_handles = legend_handles + [
        mlines.Line2D([0], [0], color="gray", linewidth=1.5, linestyle="-",  label="— BCE"),
        mlines.Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", label="-- TDQC"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=min(len(legend_handles), 4),
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined grid: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot split-CP TPR/FPR vs alpha from W&B")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity; defaults to current viewer username")
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument("--table_prefix", type=str, default="classify_cp_maxsofar", help="Table prefix to read")
    parser.add_argument("--eval_time", type=str, default="by final end", help="Filter table rows by time")
    parser.add_argument("--calib_on", type=str, default="neg", choices=["neg", "pos"], help="Filter table rows by calib on")
    parser.add_argument("--output_dir", type=str, default="notebooks/figures/cp_alpha_curves", help="Directory for outputs")
    parser.add_argument("--output_prefix", type=str, default="split_cp", help="Filename prefix")
    args = parser.parse_args()

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 19,
        "axes.titlesize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 17,
    })

    api = wandb.Api(timeout=100)
    entity = args.entity or api.viewer.username
    project_path = f"{entity}/{args.project}"

    all_runs = list(api.runs(project_path))
    if not all_runs:
        raise ValueError(f"No runs found in project '{project_path}'")

    all_rows = []
    run_count_by_method = {}
    for method_key, method_label in LABEL_MAPPING.items():
        matched_runs = find_runs_for_method_key(all_runs, method_key)
        run_count_by_method[method_key] = len(matched_runs)
        if not matched_runs:
            continue

        exp_key = infer_experiment_key(method_key)
        
        for run in matched_runs:
            df_run = extract_cp_rows(
                run=run,
                table_prefix=args.table_prefix,
                eval_time=args.eval_time,
                calib_on=args.calib_on,
            )
            if len(df_run) == 0:
                continue
            df_run = df_run.copy()
            df_run["method_key"] = method_key
            df_run["method_label"] = method_label
            df_run["experiment_key"] = exp_key
            all_rows.append(df_run)

    if not all_rows:
        raise ValueError(
            "Matched runs found, but no rows matched filters. "
            "Check table_prefix/eval_time/calib_on and available W&B table keys."
        )

    df = pd.concat(all_rows, ignore_index=True)

    # Keep only rows from split-CP logs with required plotting columns.
    required_cols = ["experiment_key", "method_key", "method_label", "alpha", "tpr", "fpr", "tnr", "run_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in extracted data: {missing}")

    group_cols = ["experiment_key", "method_key", "method_label", "alpha"]
    df_stats = df.groupby(group_cols)[["tpr", "fpr", "tnr"]].agg(["mean", "std", "count"]).reset_index()
    df_stats.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in df_stats.columns]

    os.makedirs(args.output_dir, exist_ok=True)
    raw_csv = os.path.join(args.output_dir, f"{args.output_prefix}_raw.csv")
    agg_csv = os.path.join(args.output_dir, f"{args.output_prefix}_stats.csv")
    df.to_csv(raw_csv, index=False)
    df_stats.to_csv(agg_csv, index=False)

    experiments = sorted(df_stats["experiment_key"].unique())
    for exp_key in experiments:
        df_exp = df_stats[df_stats["experiment_key"] == exp_key].copy()
        exp_prefix = f"{args.output_prefix}_{exp_key}"
        exp_title = EXPERIMENT_DISPLAY_NAMES.get(exp_key, exp_key)

        out = os.path.join(args.output_dir, f"{exp_prefix}_tpr_vs_alpha.png")
        plot_metric_vs_alpha(df_exp, metric_key="tpr", output_path=out, experiment_title=exp_title)

        out = os.path.join(args.output_dir, f"{exp_prefix}_fpr_vs_alpha.png")
        plot_metric_vs_alpha(df_exp, metric_key="fpr", output_path=out, experiment_title=exp_title)

    combined_path = os.path.join(args.output_dir, f"{args.output_prefix}_combined_grid.png")
    plot_combined_grid(df_stats, output_path=combined_path)

    print(f"Project: {project_path}")
    print(f"Total runs in project: {len(all_runs)}")
    print("Matched runs per method key:")
    for method_key in sorted(run_count_by_method):
        print(f"  {method_key}: {run_count_by_method[method_key]}")
    print(f"Experiments plotted: {experiments}")
    print(f"Saved raw rows: {raw_csv}")
    print(f"Saved aggregated stats: {agg_csv}")
    print(f"Saved figures under: {args.output_dir}")


if __name__ == "__main__":
    main()
