"""Reference recipe settings and the measured learning-rate curve."""
import json
from pathlib import Path
import torch
from train import build_config, reference_learning_rate
from fpp_libero_mem.prefix_model import PrefixPatchDiT


def test_learning_rate_matches_recorded_reference():
    root=Path(__file__).resolve().parents[1]
    records=json.loads((root/'results/libero-mem-reference/fpp.training-metrics.json').read_text())
    for row in records:
        assert abs(reference_learning_rate(row['step'])-row['lr'])<1e-16
    assert reference_learning_rate(500)==1e-4
    assert reference_learning_rate(30000)==1e-6


def test_reference_model_configuration_and_spatial_buffer():
    config=build_config({'max_observations':144},steps=20000)
    assert config['temporal_position']=='rope_spatial'
    config.update(feature_dim=12,text_dim=8,cameras=1,grid_size=2,max_steps=3,
                  visual_dim=16,visual_layers=1,visual_heads=2,decoder_dim=16,
                  decoder_layers=1,decoder_heads=2,checkpoint_visual=False)
    model=PrefixPatchDiT(config)
    assert model.visual.spatial_position.shape==(4,16)
    assert not torch.equal(model.visual.spatial_position[0],model.visual.spatial_position[1])
