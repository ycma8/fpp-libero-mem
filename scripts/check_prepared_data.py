"""Validate shapes, episode lengths, action statistics and context capacity."""
import argparse
import json
from pathlib import Path
import numpy as np


def validate(root):
    root = Path(root) / 'prepared'
    manifest = json.loads((root / 'manifest.json').read_text())
    text = np.load(root / 'text.npy')
    if text.shape != (10, 512) or not np.isfinite(text).all():
        raise ValueError('Expected finite [10,512] text features')
    if manifest['camera_keys'] != ['agentview_rgb']:
        raise ValueError('Expected single agent-view cache')
    stats = manifest['normalization']
    for key in ['center', 'scale', 'constant']:
        if np.asarray(stats[key]).shape != (7,):
            raise ValueError('Action statistics must have seven dimensions')
    if not np.isfinite(stats['center']).all() or not np.isfinite(stats['scale']).all() or min(stats['scale']) <= 0:
        raise ValueError('Invalid normalization')
    episodes = manifest['episodes']
    if not episodes or not any(e['split'] == 'train' for e in episodes):
        raise ValueError('No training episodes')
    stride = manifest['history_stride']
    if stride < 1:
        raise ValueError('Invalid stride')
    seen = set()
    for e in episodes:
        if e['cache'] in seen or e['task'] not in range(1,11) or e['split'] not in ['train','val']:
            raise ValueError('Duplicate cache, invalid task or split')
        seen.add(e['cache'])
        a = np.load(root / e['cache'] / 'actions.npy', mmap_mode='r')
        x = np.load(root / e['cache'] / 'features.npy', mmap_mode='r')
        if a.shape != (e['length'], 7) or x.shape != (e['length'],196,768) or len(a) == 0:
            raise ValueError(f'Invalid episode shape: {e["cache"]}')
        if not np.isfinite(a).all() or not np.isfinite(x).all():
            raise ValueError('Non-finite episode')
        if (len(a)+stride-1)//stride > manifest['max_observations']:
            raise ValueError('Episode exceeds context capacity')
    return {'episodes':len(episodes),'tasks':len({e['task'] for e in episodes}),'text_shape':list(text.shape)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data_root', type=Path)
    print(json.dumps(validate(p.parse_args().data_root), indent=2))
