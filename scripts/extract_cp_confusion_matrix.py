"""
Extract classification metrics from wandb table `classify_cp_maxsofar` and compute confusion matrices.

Usage:
    python scripts/extract_cp_confusion_matrix.py \
        --project tdqc --run_name openvla-widowx-q_learning-top_k_probs_test \
        [--table_key classify_cp_maxsofar/model] \
        [--eval_time "by final end"] \
        [--calib_on neg] \
        [--alpha 0.2] \
        [--output confusion_matrix_results.csv]
        
    # To list all available table keys in a run:
    python scripts/extract_cp_confusion_matrix.py \
        --project tdqc --run_name openvla-widowx-q_learning-top_k_probs_test --list_keys
"""

import argparse
import json
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd
import wandb


def load_table_from_run(run: wandb.apis.public.Run, api: wandb.Api) -> dict[str, pd.DataFrame]:
    """Load all wandb tables from a run's media files, keyed by their original logged key.

    Tables are stored under ``media/table/`` in the run's files, e.g.
    ``media/table/classify_cp_maxsofar/model_24040_622b4c8d.table.json``
    → key ``classify_cp_maxsofar/model``.
    """
    tables = {}
    tmp_dir = tempfile.mkdtemp()

    for f in run.files():
        if not f.name.startswith("media/table/") or not f.name.endswith(".table.json"):
            continue
        try:
            f.download(root=tmp_dir, replace=True)
            local_path = os.path.join(tmp_dir, f.name)
            table_json = json.load(open(local_path, "r"))
            df = pd.DataFrame(table_json["data"], columns=table_json["columns"])
            # Strip media/table/ prefix, then _<step>_<hash>.table.json suffix
            rel = f.name[len("media/table/"):].replace(".table.json", "")
            key = re.sub(r"_\d+_[a-f0-9]+$", "", rel)
            tables[key] = df
        except Exception as e:
            print(f"  Warning: failed to load file '{f.name}': {e}")

    return tables


def compute_confusion_matrix_from_rates(row: pd.Series, n_total: int = None):
    """
    Reconstruct absolute confusion-matrix counts from TPR/FPR/TNR/FNR rates.
    If n_total is unknown, counts are returned as fractional (rates themselves).
    """
    tpr = row.get("tpr", np.nan)
    tnr = row.get("tnr", np.nan)
    fpr = row.get("fpr", np.nan)
    fnr = row.get("fnr", np.nan)

    return {
        "TP_rate": tpr,
        "FN_rate": fnr,
        "FP_rate": fpr,
        "TN_rate": tnr,
    }


def print_confusion_matrix(row: pd.Series, label: str = ""):
    """Pretty-print a confusion matrix from a row of metrics."""
    tpr = row.get("tpr", 0)
    fnr = row.get("fnr", 0)
    fpr = row.get("fpr", 0)
    tnr = row.get("tnr", 0)

    header = f"  Confusion Matrix{f' ({label})' if label else ''}"
    print(header)
    print("  " + "-" * 40)
    print(f"  {'':>20} | {'Pred Pos':>10} | {'Pred Neg':>10}")
    print("  " + "-" * 40)
    print(f"  {'Actual Pos (Fail)':>20} | {'TP':>4}={tpr:>5.3f} | {'FN':>4}={fnr:>5.3f}")
    print(f"  {'Actual Neg (Succ)':>20} | {'FP':>4}={fpr:>5.3f} | {'TN':>4}={tnr:>5.3f}")
    print("  " + "-" * 40)


def find_runs_by_name(api: wandb.Api, project_path: str, run_name: str) -> list:
    """Find all runs matching a given name (exact or prefix with seed suffix)."""
    all_runs = list(api.runs(project_path))
    matched = []
    for r in all_runs:
        if r.name == run_name or (
            r.name.startswith(run_name) and
            (suffix := r.name[len(run_name):]) != "" and
            suffix.startswith("-") and suffix[1:].split("-")[0].isdigit()
        ):
            matched.append(r)
    return matched


def main():
    parser = argparse.ArgumentParser(
        description="Extract CP classification metrics and confusion matrices from wandb."
    )
    parser.add_argument("--project", type=str, required=True,
                        help="wandb project name (e.g. 'tdqc')")
    parser.add_argument("--run_name", type=str, required=True,
                        help="wandb run name (e.g. 'openvla-widowx-q_learning-top_k_probs_test'). "
                             "Matches exact name or prefix with seed suffix.")
    parser.add_argument(
        "--table_key",
        type=str,
        default=None,
        help="Table key to extract, e.g. 'classify_cp_maxsofar/model'. "
             "If omitted, extracts all classify_cp_maxsofar tables.",
    )
    parser.add_argument("--eval_time", type=str, default=None,
                        help="Filter by eval time, e.g. 'by final end', 'by earliest stop', 'at earliest stop'")
    parser.add_argument("--calib_on", type=str, default=None,
                        help="Filter by calibration label: 'pos' or 'neg'")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Filter by a specific alpha value")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results CSV. If omitted, only prints to stdout.")
    parser.add_argument("--list_keys", action="store_true",
                        help="List all available table keys in the run and exit.")
    args = parser.parse_args()

    # Connect to wandb
    api = wandb.Api(timeout=60)
    WANDB_USERNAME = api.viewer.username
    project_path = f"{WANDB_USERNAME}/{args.project}"
    print(f"Project: {project_path}")
    print(f"Searching for runs matching: '{args.run_name}'...")

    runs = find_runs_by_name(api, project_path, args.run_name)
    if not runs:
        print(f"Error: no runs found matching '{args.run_name}' in {project_path}")
        sys.exit(1)
    print(f"Found {len(runs)} matching run(s): {[r.name for r in runs]}")

    # Process each matching run
    all_results = []
    for run in runs:
        print(f"\n{'#' * 60}")
        print(f"Run: {run.name}  (id={run.id}, state={run.state})")
        print(f"{'#' * 60}")

        # Load all tables
        print("Downloading table artifacts...")
        tables = load_table_from_run(run, api)

        if args.list_keys:
            print("\nAvailable table keys:")
            for k in sorted(tables.keys()):
                print(f"  {k}  ({len(tables[k])} rows, columns: {list(tables[k].columns)})")
            continue

        # Select which tables to process
        if args.table_key:
            matched = {k: v for k, v in tables.items() if args.table_key in k}
            if not matched:
                print(f"\n  Warning: table key '{args.table_key}' not found. Available keys:")
                for k in sorted(tables.keys()):
                    print(f"    {k}")
                continue
        else:
            matched = {k: v for k, v in tables.items() if "classify_cp_maxsofar" in k}
            if not matched:
                print("\n  No classify_cp_maxsofar tables found. Available keys:")
                for k in sorted(tables.keys()):
                    print(f"    {k}")
                continue

        # Process each matched table
        for table_name, df in matched.items():
            print(f"\n{'=' * 60}")
            print(f"Table: {table_name}  ({len(df)} rows)")
            print(f"Columns: {list(df.columns)}")
            print(f"{'=' * 60}")

            # Apply filters
            filtered = df.copy()
            if args.eval_time and "time" in filtered.columns:
                filtered = filtered[filtered["time"] == args.eval_time]
            if args.calib_on and "calib on" in filtered.columns:
                filtered = filtered[filtered["calib on"] == args.calib_on]
            if args.alpha is not None and "alpha" in filtered.columns:
                filtered = filtered[(filtered["alpha"] - args.alpha).abs() < 1e-6]

            if len(filtered) == 0:
                print("  No rows match the given filters.")
                continue

            print(f"  {len(filtered)} rows after filtering.\n")

            # Define the metric columns we want
            metric_cols = ["tpr", "tnr", "fpr", "fnr", "acc", "bal_acc", "f1", "weighted-acc"]
            info_cols = [
                c for c in ["detect_method", "cal split", "test split", "calib on",
                             "task", "thresh_method", "alpha", "time", "threshold", "avg_det_time"]
                if c in filtered.columns
            ]

            for idx, row in filtered.iterrows():
                result_row = {"run_name": run.name, "table": table_name}
                for c in info_cols:
                    result_row[c] = row[c]
                for m in metric_cols:
                    result_row[m] = row.get(m, np.nan)
                cm = compute_confusion_matrix_from_rates(row)
                result_row.update(cm)
                all_results.append(result_row)

    # Save to CSV if requested
    if args.output and all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output} ({len(results_df)} rows)")
    elif args.output:
        print("\nNo results to save.")

    # Print summary table
    if all_results:
        results_df = pd.DataFrame(all_results)
        summary_cols = ["run_name", "table", "alpha", "time", "calib on", "tpr", "tnr", "fpr", "fnr",
                        "acc", "bal_acc", "f1"]
        summary_cols = [c for c in summary_cols if c in results_df.columns]
        print(f"\n{'=' * 60}")
        print("SUMMARY (per run)")
        print(f"{'=' * 60}")
        with pd.option_context("display.max_rows", None, "display.max_columns", None,
                               "display.width", 200, "display.float_format", "{:.4f}".format):
            print(results_df[summary_cols].to_string(index=False))

        # Print averaged confusion matrix across all runs
        metric_cols_avg = ["tpr", "tnr", "fpr", "fnr", "acc", "bal_acc", "f1", "weighted-acc"]
        metric_cols_avg = [c for c in metric_cols_avg if c in results_df.columns]
        group_cols = [c for c in ["table", "alpha", "time", "calib on", "thresh_method", "task"]
                      if c in results_df.columns]

        if group_cols:
            avg_df = results_df.groupby(group_cols)[metric_cols_avg].agg(["mean", "std"]).reset_index()
            # Flatten multi-level columns
            avg_df.columns = [
                f"{c[0]}_{c[1]}" if c[1] else c[0]
                for c in avg_df.columns
            ]

            print(f"\n{'=' * 60}")
            print(f"AVERAGED ACROSS {len(runs)} RUN(S)")
            print(f"{'=' * 60}")

            display_cols = group_cols + [f"{m}_mean" for m in metric_cols_avg if f"{m}_mean" in avg_df.columns]
            with pd.option_context("display.max_rows", None, "display.max_columns", None,
                                   "display.width", 200, "display.float_format", "{:.4f}".format):
                print(avg_df[display_cols].to_string(index=False))

            # Print averaged confusion matrices
            print(f"\n{'=' * 60}")
            print(f"AVERAGED CONFUSION MATRICES ({len(runs)} run(s))")
            print(f"{'=' * 60}")
            for _, arow in avg_df.iterrows():
                ctx = ", ".join(f"{c}={arow[c]}" for c in group_cols)
                tpr_m = arow.get("tpr_mean", 0)
                fnr_m = arow.get("fnr_mean", 0)
                fpr_m = arow.get("fpr_mean", 0)
                tnr_m = arow.get("tnr_mean", 0)
                tpr_s = arow.get("tpr_std", 0)
                fnr_s = arow.get("fnr_std", 0)
                fpr_s = arow.get("fpr_std", 0)
                tnr_s = arow.get("tnr_std", 0)

                print(f"\n  {ctx}")
                print("  " + "-" * 52)
                print(f"  {'':>20} | {'Pred Pos':>14} | {'Pred Neg':>14}")
                print("  " + "-" * 52)
                print(f"  {'Actual Pos (Fail)':>20} | TP={tpr_m:.3f}±{tpr_s:.3f} | FN={fnr_m:.3f}±{fnr_s:.3f}")
                print(f"  {'Actual Neg (Succ)':>20} | FP={fpr_m:.3f}±{fpr_s:.3f} | TN={tnr_m:.3f}±{tnr_s:.3f}")
                print("  " + "-" * 52)


if __name__ == "__main__":
    main()
