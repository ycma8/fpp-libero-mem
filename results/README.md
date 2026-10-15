# LIBERO-Mem result provenance

`libero_mem.csv` preserves the reported task-level counts from the existing
FPP evaluation snapshot. Each policy has 20 trials per task. The FPP result is
associated with the 20,000-update checkpoint; the displayed DP recordings
are labeled 30,000 updates. The [reference audit](libero-mem-reference/README.md)
now includes the original per-trial DP counts as well as the FPP records; all
task-level values were checked directly, rather than relying on reconstructed
percentages.

Whole-task success is total successes / 200. Mean subgoal completion is the
arithmetic mean of each task's completed / available subgoals. The pooled
414/620 and 110/620 ratios are not the reported task means.

The full-horizon evaluation protocol is documented in the root README. The
training/evaluation settings, recovered split metadata, checkpoint/source hashes
and scalar training curves are in `libero-mem-reference/`. Run
`python scripts/verify_results.py` from the repository root to recompute the
reported metrics.
