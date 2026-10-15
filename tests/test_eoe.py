"""Keep the EOE ablation separate from real-frame/action boundary handling."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fpp_libero_mem.data import Episodes, full_collate


class EOETests(unittest.TestCase):
    def test_default_terminal_supervision_and_explicit_opt_out(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory) / 'prepared'
            episode = prepared / 'episode'
            episode.mkdir(parents=True)
            features = np.arange(5*196*2, dtype=np.float32).reshape(5, 196, 2)
            actions = np.arange(35, dtype=np.float32).reshape(5, 7)
            np.save(episode/'features.npy', features)
            np.save(episode/'actions.npy', actions)
            np.save(prepared/'text.npy', np.zeros((10, 512), dtype=np.float32))
            manifest = {'camera_keys': ['agentview_rgb'], 'history_stride': 2,
                'max_observations': 4, 'normalization': {'center': [0]*7, 'scale': [1]*7},
                'episodes': [{'task': 1, 'split': 'train', 'cache': 'episode'}]}
            (prepared/'manifest.json').write_text(json.dumps(manifest))
            for phase in [0, 1]:
                with patch('numpy.random.randint', return_value=phase):
                    current = Episodes(directory, append_eoe=False)[0]
                    legacy = Episodes(directory)[0]
                np.testing.assert_array_equal(current['features'], features[phase::2])
                np.testing.assert_array_equal(legacy['features'][:-1], current['features'])
                np.testing.assert_array_equal(legacy['features'][-1], features[-1])
                self.assertEqual(len(legacy['valid']), len(current['valid'])+1)
                np.testing.assert_array_equal(current['actions'][-1, -1], actions[-1])
                np.testing.assert_array_equal(legacy['actions'][-1], np.repeat(actions[-1:], 16, axis=0))
            with patch('numpy.random.randint', return_value=0):
                longer = Episodes(directory, append_eoe=False)[0]
            with patch('numpy.random.randint', return_value=1):
                shorter = Episodes(directory, append_eoe=False)[0]
            batch = full_collate([longer, shorter])
            self.assertFalse(batch['valid'][1, -1])
            self.assertEqual(int(batch['valid'].sum()), 5)


if __name__ == '__main__':
    unittest.main()
