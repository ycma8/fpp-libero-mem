"""Evaluate an FPP checkpoint on held-out LIBERO-Mem initial states.

Requires the upstream LIBERO-Mem simulator, its assets, and HDF5 demonstrations.
The image transform is explicit because upstream dataset exports differ.
"""
import argparse
import glob
import hashlib
import json
import random
import sys
from pathlib import Path
import h5py
import numpy as np
import torch
from fpp_libero_mem.data import TASK_MIN_STEPS, strip_leading, task_number
from fpp_libero_mem.evaluation import SUBGOALS, rollout, summarize, validate_trial_plan


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def trials(args):
    files = sorted(map(Path, glob.glob(args.raw_glob)), key=task_number)
    if [task_number(p) for p in files] != list(range(1,11)):
        raise ValueError('Expected exactly ten task files numbered 1..10')
    result = []
    for path in files:
        task = task_number(path)
        bddls = list(args.bddl_root.glob(f'KITCHEN_SCENE1_{task}_*.bddl'))
        if len(bddls) != 1:
            raise ValueError(f'Expected one BDDL file for task {task}')
        with h5py.File(path) as f:
            keys = sorted(f['data'], key=lambda x:int(x.rsplit('_',1)[-1]))
            if len(keys) <= 20:
                raise ValueError('Need training episodes plus 20 held-out episodes')
            def state(key):
                g = f['data'][key]
                return np.asarray(g['init_state'] if 'init_state' in g else g['states'][0], dtype=np.float64)
            train_hashes = {hashlib.sha256(state(k).tobytes()).hexdigest() for k in keys[:-20]}
            for key in keys[-20:]:
                init = state(key)
                init_hash = hashlib.sha256(init.tobytes()).hexdigest()
                if init_hash in train_hashes:
                    raise ValueError(f'Train/test initial-state overlap: {path.name}:{key}')
                actions = np.asarray(f['data'][key]['actions'])
                horizon = max(TASK_MIN_STEPS[task-1], len(strip_leading(actions)) + 20)
                result.append({'task': task, 'episode': key, 'initial_state': init,
                               'init_sha256':init_hash, 'horizon':horizon, 'bddl':str(bddls[0]),
                               'source':path.name})
    return result


def main(args):
    plan = trials(args)
    manifest = json.loads((args.data_root/'prepared/manifest.json').read_text())
    # Require the same held-out membership used by the prepared training cache.
    validate_trial_plan(manifest, plan)
    if args.dry_run:
        print(json.dumps({'trials':len(plan), 'max_horizon':max(r['horizon'] for r in plan),
                          'tasks':10, 'holdout_membership_verified':True},indent=2))
        return
    full_trial_count = len(plan)
    if args.max_trials is not None:
        if not 1 <= args.max_trials <= full_trial_count:
            raise ValueError('--max-trials must be between one and the full trial count')
        plan = plan[:args.max_trials]
    if args.checkpoint is None or args.dino_checkpoint is None:
        raise ValueError('Provide --checkpoint and --dino-checkpoint for rollout')
    if args.output.exists():
        raise FileExistsError('Output already exists; choose a new run directory')
    import timm
    from libero.libero.envs import OffScreenRenderEnv
    from fpp_libero_mem.inference import Policy
    from scripts.prepare_cache import _features
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    text = np.load(args.data_root/'prepared/text.npy')
    if text.shape != (10,512) or not np.isfinite(text).all():
        raise ValueError('Expected finite [10,512] text features')
    device = torch.device(args.device)
    policy = Policy(args.checkpoint, device=args.device)
    manifest_sha256 = sha(args.data_root/'prepared/manifest.json')
    if policy.metadata.get('data_manifest_sha256') not in {None, manifest_sha256}:
        raise ValueError('Checkpoint and evaluation data manifests differ')
    stride = manifest['history_stride']
    if policy.metadata['history_stride'] != stride:
        raise ValueError('Checkpoint and cache strides differ')
    for key, path in [('text_features_sha256',args.data_root/'prepared/text.npy'),
                      ('dino_checkpoint_sha256',args.dino_checkpoint)]:
        if policy.metadata[key] != sha(path):
            raise ValueError(f'Checkpoint preprocessing mismatch: {key}')
    backbone = timm.create_model(manifest['dino_model'], pretrained=False,
                                 checkpoint_path=str(args.dino_checkpoint)).to(device).eval()
    encode = lambda rgb: _features(backbone, rgb[None], device, 1)[0]
    args.output.mkdir(parents=True)
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    settings.update(checkpoint_sha256=sha(args.checkpoint), simulator_revision=args.simulator_revision,
                    history_stride=stride, protocol='deterministic-xml-reset-full-horizon-v3',
                    full_trial_count=full_trial_count, requested_trial_count=len(plan),
                    data_manifest_sha256=manifest_sha256,
                    seed_rule='base_seed + 1000 * task + episode_number',
                    evaluator_sha256=sha(__file__),
                    evaluation_library_sha256=sha(Path(__file__).parent/'fpp_libero_mem/evaluation.py'))
    settings['runtime'] = dict(python=sys.version.split()[0],torch=torch.__version__,
        cuda=torch.version.cuda,numpy=np.__version__,h5py=h5py.__version__,timm=timm.__version__,
        device=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
    (args.output/'settings.json').write_text(json.dumps(settings,indent=2))
    records=[]
    for trial in plan:
        trial_seed = args.seed + 1000 * trial['task'] + int(trial['episode'].rsplit('_',1)[-1])
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        snapshots = []
        capture = args.save_frames and not any(r['task']==trial['task'] for r in records)
        last_query = (trial['horizon']-1)//stride
        frame_steps = set((np.linspace(0,last_query,6).round().astype(int)*stride).tolist())
        def observer(step,rgb):
            if capture and step in frame_steps:
                snapshots.append((step,rgb.copy()))
        env = OffScreenRenderEnv(bddl_file_name=trial['bddl'],camera_heights=args.camera_size,
                                 camera_widths=args.camera_size, camera_depths=False,
                                 camera_segmentations=None, horizon=trial['horizon']+args.settle_steps+1,
                                 ignore_done=True)
        try:
            env.seed(trial_seed)
            row = rollout(env, policy, encode, text[trial['task']-1], trial['initial_state'],
                          trial['horizon'], stride, SUBGOALS[trial['task']-1], args.image_transform,
                          args.settle_steps, args.flow_steps, observer=observer)
        finally:
            env.close()
        row.update({k:v for k,v in trial.items() if k!='initial_state'})
        row['seed'] = trial_seed
        if snapshots:
            from PIL import Image
            frames_dir = args.output/'frames'/f"task_{trial['task']:02d}_{trial['episode']}"
            frames_dir.mkdir(parents=True)
            for step,rgb in snapshots:
                Image.fromarray(rgb).save(frames_dir/f'step_{step:04d}.png')
            row['frames'] = [str((frames_dir/f'step_{step:04d}.png').relative_to(args.output))
                             for step,_ in snapshots]
        records.append(row)
        with (args.output/'trials.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        (args.output/'summary.json').write_text(json.dumps(summarize(records),indent=2))
        print(json.dumps(row),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-glob',required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--bddl-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--dino-checkpoint',type=Path)
    p.add_argument('--output',type=Path,default=Path('outputs/evaluation'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--image-transform',choices=['none','flipud','rotate180'],required=True)
    p.add_argument('--camera-size',type=int,default=256)
    p.add_argument('--flow-steps',type=int,default=10)
    p.add_argument('--settle-steps',type=int,default=20)
    p.add_argument('--simulator-revision',required=True,help='git rev-parse HEAD of installed LIBERO-Mem')
    p.add_argument('--dry-run',action='store_true',help='Validate 200 initial states without simulator or weights')
    p.add_argument('--max-trials',type=int,help='Smoke-test subset only; omit for the complete benchmark')
    p.add_argument('--save-frames',action='store_true',help='Save six uniformly spaced observations from the first trial of each task')
    args=p.parse_args()
    if args.flow_steps<1 or args.settle_steps<0 or args.camera_size<1:
        p.error('Invalid flow, settling or image size')
    main(args)
