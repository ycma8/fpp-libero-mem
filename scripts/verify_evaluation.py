"""Verify a complete fresh evaluation without simulator or checkpoint imports."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

GOALS = [1, 1, 3, 3, 5, 7, 3, 4, 2, 2]


def verify(root: Path, manifest_path: Path) -> dict:
    """Check complete trial membership and aggregate and per-task metrics.

    Returns:
        Recomputed metrics and source-file fingerprints.

    Raises:
        ValueError: If the records are incomplete or inconsistent.
    """
    settings = json.loads((root / "settings.json").read_text())
    records = [
        json.loads(line) for line in (root / "trials.jsonl").read_text().splitlines()
    ]
    manifest = json.loads(manifest_path.read_text())
    plan = {
        (r["task"], r["source"].split(":", 1)[1]): r
        for r in manifest["evaluation_trials"]
    }
    if (
        len(manifest["evaluation_trials"]) != 200
        or len(plan) != 200
        or len(records) != 200
    ):
        raise ValueError("Expected the complete 200-trial evaluation")
    if settings["full_trial_count"] != 200 or settings["requested_trial_count"] != 200:
        raise ValueError("A smoke/subset run is not a benchmark result")
    keys = {(r["task"], r["episode"]) for r in records}
    if len(keys) != 200 or keys != set(plan):
        raise ValueError("Evaluation membership differs or contains duplicate trials")
    if Counter(row["task"] for row in records) != Counter(
        {task: 20 for task in range(1, 11)}
    ):
        raise ValueError("Expected twenty trials for each of ten tasks")
    for row in records:
        expected = plan[row["task"], row["episode"]]
        if f"{row['source']}:{row['episode']}" != expected["source"]:
            raise ValueError("Trial source file differs from the frozen plan")
        if (
            row["init_sha256"] != expected["initial_state_sha256"]
            or row["steps"] != expected["horizon"]
        ):
            raise ValueError("State or horizon differs from the frozen plan")
        if row["seed"] != settings["seed"] + 1000 * row["task"] + int(
            row["episode"].rsplit("_", 1)[1]
        ):
            raise ValueError("Per-trial seed mismatch")
        total = GOALS[row["task"] - 1]
        if type(row["success"]) is not bool or type(row["overshot"]) is not bool:
            raise ValueError("Success and overshot must be boolean values")
        if (
            type(row["subgoals_completed"]) is not int
            or row["subgoals_total"] != total
            or not 0 <= row["subgoals_completed"] <= total
        ):
            raise ValueError("Subgoal count outside the protocol")
        if row["success"] and (row["overshot"] or row["subgoals_completed"] != total):
            raise ValueError("Invalid success/overshoot combination")
    successes = sum(r["success"] for r in records)
    completed = sum(r["subgoals_completed"] for r in records)
    total = sum(r["subgoals_total"] for r in records)
    task_means = [
        sum(r["subgoals_completed"] for r in records if r["task"] == t)
        / (20 * GOALS[t - 1])
        for t in range(1, 11)
    ]
    summary = json.loads((root / "summary.json").read_text())
    recomputed = dict(
        trials=200,
        successes=successes,
        whole_task_success=successes / 200,
        subgoals_completed=completed,
        subgoals_total=total,
        mean_subgoal_completion=sum(task_means) / 10,
        pooled_subgoal_completion=completed / total,
        overshot=sum(r["overshot"] for r in records),
    )
    for key, value in recomputed.items():
        if not math.isfinite(summary[key]) or abs(summary[key] - value) > 1e-10:
            raise ValueError(f"Summary differs from trial records: {key}")
    tasks = summary.get("tasks", [])
    if len(tasks) != 10 or {row["task"] for row in tasks} != set(range(1, 11)):
        raise ValueError("Per-task summary is incomplete")
    for task_summary in tasks:
        task = task_summary["task"]
        rows = [row for row in records if row["task"] == task]
        if (
            task_summary["trials"] != 20
            or task_summary["successes"] != sum(row["success"] for row in rows)
            or not math.isfinite(task_summary["subgoal_completion"])
            or abs(task_summary["subgoal_completion"] - task_means[task - 1]) > 1e-10
        ):
            raise ValueError("Per-task summary differs from trial records")
    return dict(
        verified=True,
        **recomputed,
        trials_sha256=hashlib.sha256((root / "trials.jsonl").read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.evaluation, args.manifest), indent=2))
