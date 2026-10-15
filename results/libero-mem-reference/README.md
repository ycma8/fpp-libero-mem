# LIBERO-Mem reference experiment

This directory contains training settings, source/checkpoint hashes and 400
per-trial records for the FPP and two-frame DP experiments on LIBERO-Mem.
Prepared-data manifests and replay records are identified by the hashes stored
in the training and evaluation configurations.

## Results

| Policy | Whole-task success | Task-mean subgoal completion | Pooled subgoals |
|---|---:|---:|---:|
| FPP | 101/200 (50.50%) | 68.55% | 414/620 (66.77%) |
| DP, two frames | 5/200 (2.50%) | 17.99% | 110/620 (17.74%) |

Each policy has 20 trials per task, with identical task/initial-state IDs,
horizons, split and seed across policies. No duplicate trial IDs were found.
Both have 17 overshot rollouts. Whole-task success requires final `done`, all
ordered subgoals completed, and no overshoot. The evaluator runs the full
horizon rather than stopping at an intermediate `done`.

The task mean gives each of ten tasks equal weight. The pooled fraction gives
each subgoal opportunity equal weight. They are different statistics and must
not be interchanged in comparisons to published results.

From the repository root, recompute the table with Python's standard library:

```bash
python scripts/verify_results.py
```

Expected output is `"status": "passed"`, followed by aggregate and task-level
counts. The verifier checks exported file hashes, duplicate/paired trials,
success predicates, subgoal denominators and the complete saved summary.

## Training and inputs

Training took place on H38. The FPP launch records a single GPU; H38 currently
reports RTX 4090 D GPUs. The databases record 861 training and 100 validation
demonstrations for both policies, with a common data-manifest hash and seed 42.

| Setting | Evaluated FPP | Evaluated DP |
|---|---|---|
| Evaluated update | 20,000 | 30,000 |
| Input | One agent-view RGB camera, 224×224; historical observations | Same camera/resolution; two adjacent control frames |
| History | Every 5 steps at 20 Hz (4 Hz), capacity 144 | Two consecutive 20-Hz frames |
| Visual backbone | Frozen DINOv3 ViT-B/16, 196 patches | Trainable ImageNet ResNet18, GroupNorm |
| Text | Frozen CLIP ViT-B/32 projected text feature, 512D | Same frozen text feature |
| Position features | Temporal RoPE + fixed 2D spatial sinusoids + learned camera embedding | Not applicable |
| Trainable parameters | 73,423,367 | 98,679,111 |
| Action | Native 7D OSC_POSE delta controls | Same |
| Chunk / executed steps | 16 / 5 | 16 / 5 |
| Effective batch | 8 complete histories | 32 sampled query times |
| AdamW betas / weight decay | (0.9, 0.95) / 0.01 | (0.95, 0.999) / 1e-6 |
| Base learning rate | 1e-4, scheduled | 1e-4, scheduled |
| Gradient clipping | 1.0 | 1.0 |
| Evaluated weights | Model | EMA, power 0.75 |
| Sampling | 10 Euler flow steps; visual KV cache | 100 DDPM steps |

The FPP encoder has six temporal layers and an eight-layer action head, both
width 512. The decoder reads the task prefix and current contextualized patch
block. It does not directly cross-attend to every historical block. The DP
decoder uses ConditionalUnet1D with widths [256, 512, 1024], kernel 5 and a
256D diffusion embedding. No robot state, counters, object labels, or simulator
state are supplied as policy observations.

Historical preparation maps exact-zero gripper commands to open (-1), uses
train-only symmetric min/max normalization, and restores constant dimensions
on denormalization. Each sampled frame supervises consecutive future control
actions. End-of-episode action padding repeats the final action, and this
historical FPP run also appended one valid terminal observation/hold chunk.

The original schedule warms up linearly for 500 updates and then uses cosine
decay with a 0.1 multiplier floor over a 30K budget. FPP was resumed at updates
7,000 and 7,086 with model, optimizer and main-process RNG state. Each continuation
uses cosine decay from its saved learning rate to 1e-6 at 30K. Exact boundary
values and the formula are in `experiment.json`; the recorded scalar curves
are in `fpp.training-metrics.json` and `dp.training-metrics.json`.

The batches, pretraining, training exposure and compute differ between policies.
This is a whole-policy comparison, not an isolated memory ablation or a
compute-matched benchmark.

## Environment and scoring

The historical protocol pins
[LIBERO-Mem](https://github.com/libero-mem/libero-mem) at
`0eee0defa4024e51511b6e15a79f30e9620209de`, the
[RGB dataset](https://huggingface.co/datasets/libero-mem/LIBERO-Mem) at
`6df130c05b82e09294fa9534e84a069687b72d33`, and the
[raw actions/initial states](https://huggingface.co/datasets/libero-mem/LIBERO-Mem-Raw)
at `61917674055e8b1cbbfd22c1b54c28d20c1a1584`. These are recorded source pins;
the vendored environment is a directory snapshot, not a separate Git checkout.
Evaluation source hashes are included in `experiment.json`.

Evaluation uses EGL, a vertical flip of rendered RGB to match recorded images,
and the collection-compatible XML/simulator reset sequence. Each trial creates
a new environment, restores its initial state, clears progress/overshoot state,
and takes 20 open-gripper settling steps. The evaluator increments the original
subgoal predicate once per control step. Task horizons follow the original
task-specific minimum or stripped demonstration length plus 20, whichever is
larger; the same horizons are used for both policies.

Earlier reset-protocol results were superseded and are not used here. The
corrected replay audit recorded 28/30 successful demonstration replays, with
two disclosed exceptions: T3/demo_3 exceeds the unchanged 30-mm `On` threshold
by about 0.58 mm; T8/demo_1 fails its final open-loop transport although
recorded-state replay succeeds. The evaluation configs record these exceptions,
no removed test trials and no relaxed success criteria. This audit checked the
configs and guard source; it did not rerun those demonstration replays.

## Evidence and limitations

- `fpp.rollouts.jsonl` and `dp.rollouts.jsonl` preserve the original per-trial
  metric fields. Absolute checkpoint paths are replaced with logical IDs;
  timings are omitted. DP's four shards are merged and records are sorted.
- `summary.json` is recomputed from those records. DP subgoal counts now come
  directly from rollout logs, rather than being reconstructed from percentages.
- `experiment.json` records settings, both checkpoint SHA256 hashes, the
  manifest hash, training/evaluation source hashes and the audit's limits.
- All nine archived FPP training source hashes and all nine evaluation source
  hashes matched their corresponding launch records. Both checkpoint files
  were present on H38 and their SHA256 hashes were computed during this audit.
- `checksums.json` records original artifact hashes and exported-file hashes.
  This detects accidental differences; it is not independent authentication.
- The recovered manifest has SHA256
  `7120f3640ee237e707d35089ede0136aabe3fab78e048c1869972b1b179ce9eb`.
  Its recorded train/test IDs and initial-state hashes do not overlap; 861
  train and 100 validation records agree with MLflow. `split-manifest.json`
  exports the relevant membership/hash fields without machine paths.
  This checks the recorded split; it is not a new byte-by-byte audit of every
  raw demonstration or a check for broader scene/trajectory similarity.
- `replay-audit.json` is the recovered 30-replay record, matching the original
  recorded SHA256 `a447be5467c5c83186415ecc24b1c899457466c11dc13c69b36432a3115b1492`.
  Its 28 successes and two failures are preserved; no replay was rerun here.
- Only one training seed was evaluated. The 20K checkpoint was selected after
  inspecting multiple checkpoints on this test suite. Treat these as exploratory
  results, not an untouched final test or a multi-seed estimate.

## Relation to the current package

The reported results use the 20K checkpoint described above. The package uses
temporal RoPE plus fixed 2D spatial positions, stride 5, symmetric normalization
and AdamW betas (0.9, 0.95). The training command uses the reference split and
recorded learning-rate curve. The package applies that curve in one process;
the original run used continuations, so worker/prefetch state can differ.

The public loader retains its single-camera cache format. Tests cover default
EOE supervision, an explicit opt-out, spatial RoPE/cache equivalence and the
complete recorded LR curve. Metadata serialization differs from the original
dual-cache storage even when selected model inputs agree. The original audit's
`experiment.json` records the package differences at audit time; this README
describes the restored defaults. Raw evidence files remain unchanged.

Additional experiments are preserved in their respective result directories.
