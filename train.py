"""Train the simulation-only FPP model on a prepared LIBERO-Mem cache.

The cache contains frozen DINOv3 patch features and frozen task text features;
the policy trains only the temporal visual encoder and flow-matching action
decoder. Raw demonstrations and checkpoints are intentionally outside this
repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def build_config(manifest: dict, *, steps: int, temporal_position: str = 'rope_spatial') -> dict:
    """Build the released LIBERO-Mem FPP model configuration."""
    return {
        "feature_dim": 768,
        "text_dim": 512,
        "cameras": 1,
        "grid_size": 14,
        "max_steps": int(manifest.get("max_observations", 144)),
        "visual_dim": 512,
        "visual_layers": 6,
        "visual_heads": 8,
        "action_dim": 7,
        "horizon": 16,
        "decoder_dim": 512,
        "decoder_layers": 8,
        "decoder_heads": 8,
        "ffn_ratio": 4,
        "checkpoint_visual": True,
        "temporal_position": temporal_position,
        "training_steps": steps,
    }


def _move_batch(batch: dict, device: torch.device) -> dict:
    """Move tensor fields to the selected training device."""
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def reference_learning_rate(step: int, base_lr: float = 1e-4) -> float:
    """Reproduce the measured reference run's warmup and continuation curve."""
    def initial(index: int) -> float:
        ratio = (min(index / 500, 1.) if index <= 500 else
                 .1 + .9 * .5 * (1 + math.cos(math.pi * (index - 500) / 29500)))
        return base_lr * ratio

    def continuation(index: int, start: int, first: float) -> float:
        progress = min(1., max(0., (index - start) / (30000 - start)))
        return 1e-6 + (first - 1e-6) * .5 * (1 + math.cos(math.pi * progress))

    if step <= 7000:
        return initial(step)
    first = initial(7000)
    if step <= 7086:
        return continuation(step, 7000, first)
    return continuation(step, 7086, continuation(7086, 7000, first))


def main(args: argparse.Namespace) -> None:
    """Run bounded FPP training and write model and optimizer checkpoints."""
    from fpp_libero_mem.data import Episodes, full_collate
    from fpp_libero_mem.prefix_model import PrefixPatchDiT
    from fpp_libero_mem.distributed import initialize, global_objective

    if min(args.steps, args.batch, args.log_every, args.save_every, args.action_microbatch) < 1 or args.workers < 0:
        raise ValueError('Invalid step, batch, worker or logging settings')
    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    manifest_bytes = (data_root / "prepared/manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    model_config = build_config(manifest, steps=args.steps, temporal_position=args.temporal_position)
    context = initialize(args.device, args.batch)
    device = context.device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    model_config["append_eoe"] = args.append_eoe
    dataset = Episodes(data_root, split="train", mode="full", camera_mode="agentview",
                       append_eoe=args.append_eoe)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=context.local_batch,
        sampler=torch.utils.data.RandomSampler(
            dataset, replacement=True, num_samples=args.steps * context.local_batch
        ),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=full_collate,
        persistent_workers=args.workers > 0,
    )
    raw_model = PrefixPatchDiT(model_config).to(device)
    model = (DistributedDataParallel(raw_model,
             device_ids=[device.index] if device.type == 'cuda' else None,
             broadcast_buffers=False) if context.world_size > 1 else raw_model)
    # Initialize identical model weights, then use independent rank streams.
    if context.world_size > 1:
        random.seed(args.seed + context.rank)
        np.random.seed(args.seed + context.rank)
        torch.manual_seed(args.seed + context.rank)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(0.9, args.adam_beta2)
    )
    training = dict(global_batch=args.batch, per_rank_batch=context.local_batch,
        world_size=context.world_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, adamw_betas=[0.9, args.adam_beta2],
        gradient_clip=args.gradient_clip, learning_rate_schedule=args.lr_schedule,
        schedule_budget=30000 if args.lr_schedule == 'reference' else None,
        optimizer_updates=args.steps, source_revision=args.source_revision,
        action_microbatch=args.action_microbatch,
        loss_reduction='mean over all valid observation/action targets across ranks',
        rank_seed_rule='same model initialization, then base_seed + rank')
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if context.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'config.json').exists() or (output / 'metrics.jsonl').exists():
            raise FileExistsError('Choose a fresh output directory; training does not resume implicitly')
        (output / "config.json").write_text(
            json.dumps({"model": model_config, "data_root": str(data_root), "seed": args.seed,
                        "training": training, "data_manifest_sha256": manifest_sha256}, indent=2),
            encoding="utf-8")
    if context.world_size > 1:
        dist.barrier()
    metrics_path = output / "metrics.jsonl"
    iterator = iter(loader)
    started = time.monotonic()
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = _move_batch(next(iterator), device)
        if device.type == "cpu":
            batch["features"] = batch["features"].float()
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
        with autocast:
            loss = model(
                batch["features"],
                batch["text"],
                batch["actions"],
                batch["valid"],
                action_microbatch=args.action_microbatch,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
        objective, reported_loss = global_objective(loss, batch['valid'])
        objective.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f'non-finite gradient at step {step}')
        lr = (reference_learning_rate(step, args.learning_rate)
              if args.lr_schedule == 'reference' else args.learning_rate)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        if context.rank == 0 and (step == 1 or step % args.log_every == 0 or step == args.steps):
            record = {
                "step": step,
                "loss": float(reported_loss.cpu()),
                "grad_norm": float(grad_norm),
                "lr": lr,
                "elapsed_s": time.monotonic() - started,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if context.rank == 0 and (step % args.save_every == 0 or step == args.steps):
            torch.save(
                {"format": "fpp_libero_mem_v1", "step": step, "config": model_config,
                 "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                 "normalization": manifest["normalization"], "history_stride": manifest["history_stride"],
                 "text_features_sha256": manifest.get("text_features_sha256"),
                 "dino_checkpoint_sha256": manifest.get("dino_checkpoint_sha256"),
                 "seed": args.seed, "training": training,
                 "data_manifest_sha256": manifest_sha256},
                output / f"step_{step:06d}.pt",
            )
    if context.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/libero-mem"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=8, help="Global episode batch across all ranks")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--append-eoe", action=argparse.BooleanOptionalAction, default=True,
                        help="Reference terminal observation and hold chunk (enabled by default)")
    parser.add_argument('--temporal-position', choices=['rope_spatial','rope','absolute'], default='rope_spatial')
    parser.add_argument('--lr-schedule', choices=['reference','constant'], default='reference',
                        help='Measured reference 30K LR curve including the 7K/7086 continuation boundaries')
    parser.add_argument('--adam-beta2', type=float, default=.95)
    parser.add_argument("--source-revision", help="Source Git commit recorded in run/checkpoint provenance")
    parser.add_argument("--action-microbatch", type=int, default=16,
                        help="Valid observation targets per action-head call; the global loss is unchanged")
    main(parser.parse_args())
