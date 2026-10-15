# Comparison protocol: what is and is not matched

Reader: someone assessing the fresh no-EOE FPP experiment. The goal is to make
the comparison scope explicit alongside its measured results.

## Published reference

Source: [LIBERO-Mem, Table 4](https://arxiv.org/pdf/2511.11478). This table reports
average subgoal completion, not whole-task success. Its published averages are
5.0% for π0 (one input frame), 0.0% for SlotVLA (one input frame), 5.0% for SlotVLA (eight input frames),
and 14.8% for Naive E-SlotSSM. Missing whole-task scores are not treated as zero.

The paper describes the Naive variant as using additional object/interaction
information. This is not the same supervision as FPP's image/text/action training.
We do not claim matched inputs, training compute, pretraining, or strict superiority.

The transcribed task values are in [table4-published.csv](table4-published.csv),
from arXiv version 3. Their displayed precision is retained; no trial-level counts
are inferred. In particular, published percentages alone do not recover the
underlying denominators or establish that subgoal definitions match this snapshot.

## Recovered source audit

A previously cloned copy of `libero-mem/naive-e-slotssm` was recovered at commit
`a41e2726ad6b1392c25d3caefc41da8b0013790f`. Its README describes three stages:
object-slot pretraining, SlotSSM memory pretraining, and action-head training.

`vla-scripts/finetune-3_slotssm_to_action.py` includes object bounding boxes,
segmentation, interaction counts, and derived subgoal states. It selects the
primary camera. Its defaults and the README's launch example differ, including
batch and learning-rate values. A default is not evidence of the settings that
produced a published result.

The recovered `experiments/robot/libero/run_libero_eval.py` has LIBERO-Goal
defaults; it is not sufficient to reconstruct Table 4. Separately, the pinned
LIBERO-Mem simulator's `scripts/mem_6_run_evaluation_env_pred.py` supports seen
and unseen initial states, uses the last twenty episode IDs in unseen mode,
advances ordered progress before an action, and continues to a full horizon.
The new FPP evaluator follows these explicit predicate/termination conventions.

No checkpoint hashes, exact train/test manifest, and run logs linking each
Table 4 row to that source were recovered. Repository recovery therefore does
not establish a controlled comparison. Public repository access could not be
confirmed during this audit; the recovered clone is evidence of code, not of
its present availability or the paper's exact execution.

## New FPP experiment

- One frozen agent-view DINOv3 stream and a frozen task-text feature; no oracle
  object locations, bounding boxes, counts, or subgoal labels enter the policy.
- Recovered, explicit 861-training / 100-validation membership. The 200 evaluation
  IDs and initial-state hashes are separate from training. This is the release's
  concrete split, not an assertion that it matches an unrecovered Table 4 manifest.
- No appended terminal observation. Fixed final checkpoint at 20,000 updates,
  seed 42, global batch eight. No fresh DP or EOE-control training.
- Simulator snapshot with recorded upstream pin
  `0eee0defa4024e51511b6e15a79f30e9620209de`, explicit XML/state reset, image
  orientation, settling, full horizon, and overshoot handling. The snapshot
  has no independent `.git`; actual source hashes are archived rather than
  presenting the parent repository's HEAD as the simulator revision.
- Each trial independently seeds Python, NumPy, Torch, and the environment using
  `42 + 1000 * task + episode_number`. The exact seed is recorded in each outcome.
- Whole-task success, task-mean subgoal completion, and pooled subgoal completion
  are reported separately. Only task-mean completion is placed alongside the
  published average, and always labeled **cross-paper reference**.

The experiment completed all 200 outcomes: 51/200 whole-task successes,
49.14% task-mean subgoal completion, and 297/620 (47.90%) pooled completion.
The archived 50.5% result is not a score for this training recipe. Differences
from the old run include sampling frequency, action normalization, spatial
position encoding, optimizer schedule, and EOE; this is not a one-factor EOE ablation.
