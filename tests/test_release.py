import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from fpp_libero_mem.data import Episodes, full_collate, normalize, denormalize
from fpp_libero_mem.evaluation import rollout, summarize, validate_trial_plan
from fpp_libero_mem.inference import Policy
from fpp_libero_mem.prefix_model import PrefixPatchDiT
from scripts.check_prepared_data import validate
from test_prefix_model import small_config


class ReleaseTests(unittest.TestCase):
    def test_single_camera_cache_and_shapes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'prepared'; ep=root/'task_01/demo_0'; ep.mkdir(parents=True)
            np.save(ep/'features.npy', np.ones((11,196,768),dtype=np.float16))
            np.save(ep/'actions.npy', np.ones((11,7),dtype=np.float32))
            np.save(root/'text.npy',np.zeros((10,512),dtype=np.float32))
            stats={'center':[0]*7,'scale':[1]*7,'constant':[False]*7}
            m={'normalization':stats,'camera_keys':['agentview_rgb'], 'history_stride':10,
               'max_observations':3,'episodes':[{'task':1,'split':'train','cache':'task_01/demo_0','length':11}]}
            (root/'manifest.json').write_text(json.dumps(m))
            validate(Path(d))
            item=Episodes(d)[0]
            self.assertEqual(item['features'].shape[1:],(196,768))
            self.assertLessEqual(full_collate([item])['features'].shape[1],3)
            np.save(ep/'features.npy',np.ones((11,98,768),dtype=np.float16))
            with self.assertRaises(ValueError):validate(Path(d))

    def test_action_normalization_roundtrip_and_constant(self):
        stats={'center':[1,3],'scale':[2,1e-4],'constant':[False,True]}
        a=np.array([[2,3],[5,3.]])
        np.testing.assert_allclose(denormalize(normalize(a,stats),stats),a)
        self.assertEqual(denormalize(np.ones((1,2)),stats)[0,1],3)

    def test_checkpoint_inference_reset(self):
        with tempfile.TemporaryDirectory() as d:
            config=small_config();config['temporal_position']='rope'
            m=PrefixPatchDiT(config)
            f=Path(d)/'model.pt'
            torch.save({'config':config,'model':m.state_dict(),'normalization':{
                'center':[0]*7,'scale':[1]*7,'constant':[False]*7}},f)
            p=Policy(f,device='cpu')
            a=p.predict(np.ones((8,12),dtype=np.float32),np.ones(8,dtype=np.float32),2)
            self.assertEqual(a.shape,(4,7));self.assertTrue(np.isfinite(a).all())
            self.assertEqual(p.cache.step,1);p.reset();self.assertEqual(p.cache.step,0)

    def test_rollout_does_not_stop_on_intermediate_done_or_overshoot(self):
        class Env:
            def __init__(self,overshoot):
                self.env=self;self.overshoot=overshoot;self.n=0;self.calls=[]
                self.sim=SimpleNamespace(model=SimpleNamespace(get_xml=lambda:'xml'),
                    get_state=lambda:np.zeros(2),reset=lambda:self.calls.append('sim.reset'),
                    set_state_from_flattened=lambda s:self.calls.append('sim.state'),
                    forward=lambda:self.calls.append('sim.forward'))
            def reset(self):self.n=0;self.calls.append('reset')
            def reset_from_xml_string(self,xml):self.calls.append('xml')
            def set_init_state(self,state):return {'agentview_image':np.zeros((2,2,3),dtype=np.uint8)}
            def reset_subgoal_progress(self):self._satisfied_subgoals=[]
            def step(self,a):
                self.n+=1
                self._satisfied_subgoals=[1] if self.n<3 else [1,2]
                self._overshot=self.overshoot and self.n==4
                return self.set_init_state(None),0,True,{}
            def _check_success(self,inc=False):
                assert inc
                self.calls.append(f'predicate-before-{self.n}')
                return True
        p=SimpleNamespace(config={'horizon':2,'max_steps':2},reset=lambda:None,
                          predict=lambda *a:np.zeros((2,7)))
        for overshoot in [False,True]:
            env=Env(overshoot)
            row=rollout(env,p,lambda x:x,None,None,4,2,2,'none',0)
            self.assertEqual(env.n,4)
            self.assertEqual(row['success'],not overshoot)
            self.assertEqual(row['subgoals_completed'],2)
            self.assertEqual(env.calls[:5],['reset','xml','sim.reset','sim.state','sim.forward'])
            self.assertEqual([x for x in env.calls if x.startswith('predicate')],
                             [f'predicate-before-{n}' for n in range(4)])

    def test_explicit_test_plan_and_validation_are_separate(self):
        manifest={'episodes':[{'task':1,'source':'file:train','split':'train'},
                              {'task':1,'source':'file:val','split':'val'}],
                  'evaluation_trials':[{'task':1,'source':'file:test',
                                         'initial_state_sha256':'hash','horizon':200}]}
        plan=[{'task':1,'source':'file','episode':'test','init_sha256':'hash','horizon':200}]
        validate_trial_plan(manifest,plan)
        plan[0]['init_sha256']='changed'
        with self.assertRaises(ValueError):validate_trial_plan(manifest,plan)

    def test_task_mean_is_not_pooled(self):
        rows=[dict(task=1,episode='a',success=True,subgoals_completed=1,subgoals_total=1,overshot=False),
              dict(task=2,episode='a',success=False,subgoals_completed=0,subgoals_total=7,overshot=False)]
        self.assertEqual(summarize(rows)['mean_subgoal_completion'],.5)
        with self.assertRaises(ValueError):summarize(rows+rows)
