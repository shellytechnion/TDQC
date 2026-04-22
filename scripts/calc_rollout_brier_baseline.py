from pathlib import Path
from collections import defaultdict
import re

import os

_REPO_ROOT = Path(os.environ.get("TDQC_REPO_ROOT", Path(__file__).resolve().parent.parent))

bench_roots = {
    "OpenVLA-LIBERO": _REPO_ROOT / "openvla/rollouts/single-foward/libero_10",
    "OpenVLA-WidowX": _REPO_ROOT / "openvla/rollouts/single-foward/openvla_widowx",
    "pi0-LIBERO": _REPO_ROOT / "openpi/rollouts/pi0-libero_10/env_records",
    "pi0-FAST-LIBERO": _REPO_ROOT / "openpi/rollouts/pi0fast-libero_10/env_records",
    "pi0-FAST-Droid": _REPO_ROOT / "openpi/rollouts/pi0fast_droid_0510_all/rollouts_all",
    "UniVLA-LIBERO": _REPO_ROOT / "UniVLA/rollouts/libero_10/eval/env_records",
}

patterns = [
    re.compile(r"(?P<task>.+?)--ep(?P<ep>\d+)--succ(?P<succ>[01])"),
    re.compile(r"(?P<task>.+?)_ep(?P<ep>\d+)_succ(?P<succ>[01])"),
]


def parse_success_from_name(name: str):
    for pat in patterns:
        m = pat.search(name)
        if m:
            return m.group("task"), int(m.group("ep")), int(m.group("succ"))
    return None


def compute_for_root(root: Path):
    episodes = {}
    conflicts = 0
    scanned = 0

    for fp in root.rglob("*succ*"):
        if not fp.is_file():
            continue
        scanned += 1

        parsed = parse_success_from_name(fp.name)
        if not parsed:
            continue

        task, ep, succ = parsed

        # Keep benchmark-specific task scope to avoid collisions in nested folders.
        rel_parent = fp.parent.relative_to(root)
        parent_key = str(rel_parent) if str(rel_parent) != "." else ""
        task_key = f"{parent_key}::{task}" if parent_key else task
        episode_key = (task_key, ep)

        prev = episodes.get(episode_key)
        if prev is not None and prev != succ:
            conflicts += 1
            continue

        episodes[episode_key] = succ

    if not episodes:
        return None

    per_task_successes = defaultdict(list)
    for (task_key, _ep), succ in episodes.items():
        per_task_successes[task_key].append(succ)

    task_rates = {task: sum(vals) / len(vals) for task, vals in per_task_successes.items()}
    avg_task_success_rate = sum(task_rates.values()) / len(task_rates)

    brier_vs_avg_task_sr = sum((succ - avg_task_success_rate) ** 2 for succ in episodes.values()) / len(episodes)

    return {
        "n_tasks": len(task_rates),
        "n_episodes": len(episodes),
        "avg_task_success_rate": avg_task_success_rate,
        "brier_vs_avg_task_success_rate": brier_vs_avg_task_sr,
        "scanned_files_with_succ": scanned,
        "conflicts_skipped": conflicts,
    }


results = {}
for bench, root in bench_roots.items():
    if not root.exists():
        results[bench] = {"error": f"missing path: {root}"}
        continue
    out = compute_for_root(root)
    if out is None:
        results[bench] = {"error": f"no parseable succ files under: {root}"}
    else:
        results[bench] = out

print("\nPer-benchmark worst-case baseline (constant prediction = avg task success rate):")
print("benchmark\tn_tasks\tn_episodes\tavg_task_success_rate\tbrier_vs_avg_task_success_rate")
for bench, r in results.items():
    if "error" in r:
        print(f"{bench}\tERROR\tERROR\tERROR\tERROR\t{r['error']}")
    else:
        print(
            f"{bench}\t{r['n_tasks']}\t{r['n_episodes']}\t"
            f"{r['avg_task_success_rate']:.6f}\t{r['brier_vs_avg_task_success_rate']:.6f}"
        )
