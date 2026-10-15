# Packaged-code EOE comparison

**Historical, superseded plan.** The EOE-control pilot and its paired evaluation
queue were stopped. The subsequent simplified no-EOE run completed; see
`no-eoe-20k/README.md` for that separate record. The public defaults and main
reported results have since returned to the EOE reference recipe, documented in
`libero-mem-reference/README.md`. This plan does not describe current defaults.

This is a new experiment using the GitHub training implementation. Its purpose
is to measure the effect of the extra terminal observation. It does not yet
have a success rate and must not replace the historical reference result.

Both arms start from scratch with seed 42, batch 8, a fixed 20,000-update
budget, the same prepared cache and the same evaluator. The no-EOE arm uses the
default; the legacy arm adds `--append-eoe`. Only this supervision choice differs
between arms. Intermediate checkpoints are not selected using test success.

The cache reuses the original frozen agent-view features and CLIP text features,
with the recovered 861/100 train/validation split. The original 200 test
initial states are declared separately and used by the common evaluation plan.
No test RGB is added to training. Agent-view history stride is 10 at 20 Hz
(2 Hz), capacity 144, with 196 patches at 224×224. Actions use the package's
train-minimum/range normalization. Temporal RoPE, the six/eight-layer width-512
model, constant 1e-4 AdamW learning rate, default betas and 0.01 weight decay
follow the packaged trainer. These are intentionally not presented as the
historical 4-Hz, spatial-position, scheduled-LR recipe.

The common simulator protocol uses collection-compatible XML reset, 20 settling
steps, vertical-flip RGB alignment, ten Euler integration steps, a full fixed
horizon, and replanning every ten control steps. Report strict whole-task
success, task-mean subgoal completion and overshoot for all 200 paired trials.

Initial hardware measurement on a single H38 RTX 4090 D used batch 8 and 64
observations per episode. Five measured optimizer steps after warmup had a
median 1.70 seconds/step and peak allocated memory 8.66 GiB. This is a synthetic
throughput measurement, not training evidence. Budget roughly 10–12 hours per
20K run, plus data I/O, compilation and evaluation. Actual throughput depends on
episode lengths and storage. One converted agent-view cache needs about 96.8 GB
(90.1 GiB); both arms reuse it. Saving every 5K limits checkpoint storage.

EOE is one extra sampled final frame and a hold-action target, not a special
token or a change in parameter count. Its removal reduces valid terminal
supervision and can change stopping behavior, but the direction and magnitude
remain to be measured. A single paired seed is a pilot; it does not measure
training-seed uncertainty.
