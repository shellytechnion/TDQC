"""
Extract memory/compute metrics from wandb runs by run name and compute mean ± std.

Metrics:
  - vram_reserved_mb    : per-epoch (summary = last epoch value)
  - num_params_M        : logged once at init
  - train_wall_clock_sec: logged once at end of training
  - peak_vram_mb        : logged once at end of training
  - epoch_time_sec      : per-epoch (summary = last epoch value)

Mean and std are computed across seeds (multiple runs sharing the same display name).
"""
import wandb
import pandas as pd

WANDB_USERNAME = wandb.Api().viewer.username
PROJECT_NAME = f"{WANDB_USERNAME}/tdqc-final"

RUN_NAMES = [
    "openvla-10-indep-mlp_TD0_time_log",
    "openvla-10-lstm-lstm_TD0_time_log",
    "openvla-10-indep-mlp_time_log",
    "openvla-10-lstm-lstm_time_log",
    "openvla-10-indep-mlp_BCE_time_log",
    "openvla-10-lstm-lstm_TD0_top_k_probs_GRU",
    "openvla-10-lstm-lstm_BCE_top_k_probs_GRU" 

    "pizero_fast-default-indep-mlp_BCE_time_log",
    "pizero_fast-default-indep-mlp_TD0_time_log",
    "pizero_fast-default-lstm-lstm_time_log",
    "pizero_fast-default-lstm-lstm_TD0_time_log",
    "pizero_fast-default-lstm-lstm_top_k_probs_BCE_time_log",
    "pizero_fast-default-lstm-lstm_top_k_probs_TD0_time_log",
    "pizero_fast-default-lstm-lstm_BCE_with_TDloss_time_log",
]

METRICS = [
    "vram_reserved_mb",
    "num_params_M",
    "train_wall_clock_sec",
    "peak_vram_mb",
    "epoch_time_sec",
]


def main():
    api = wandb.Api()

    # Fetch all runs and filter client-side (server-side display_name regex is unreliable)
    print(f"Fetching runs from {PROJECT_NAME}...")
    all_runs = api.runs(PROJECT_NAME, per_page=1000)
    runs = [r for r in all_runs if r.name in RUN_NAMES]
    print(f"Found {len(runs)} matching runs out of the fetched set.")

    if not runs:
        print("No matching runs found. Verify that runs with these names exist in wandb.")
        return

    records = []
    for run in runs:
        summary = run.summary._json_dict
        row = {"run_name": run.name, "run_id": run.id}
        for metric in METRICS:
            row[metric] = summary.get(metric, None)
        records.append(row)

    df = pd.DataFrame(records)
    print(f"\nRaw data ({len(df)} runs):")
    print(df[["run_name"] + METRICS].to_string(index=False))

    # Mean and std per run name across seeds
    grouped = df.groupby("run_name")[METRICS]
    mean_df = grouped.mean().rename(columns={m: f"{m}_mean" for m in METRICS})
    std_df = grouped.std(ddof=1).rename(columns={m: f"{m}_std" for m in METRICS})
    n_df = grouped.size().rename("n_seeds")

    summary_df = pd.concat([mean_df, std_df, n_df], axis=1)

    # Interleave mean/std columns per metric
    ordered_cols = []
    for m in METRICS:
        ordered_cols += [f"{m}_mean", f"{m}_std"]
    summary_df = summary_df[ordered_cols + ["n_seeds"]]

    # Preserve the requested row order
    present = [n for n in RUN_NAMES if n in summary_df.index]
    summary_df = summary_df.reindex(present)

    save_path = "scripts/memory_metrics_summary.csv"
    summary_df.to_csv(save_path, float_format="%.2f")
    print(f"\nSaved to {save_path}")
    print(summary_df.to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
