"""Matched full π0 control for the full π0.5 ContextFlow experiment."""

import dataclasses

from openpi.training import config_full_contextflow
from openpi.training.full_backbone_weights import FullBackboneWeightLoader


def build(api):
    pi05 = config_full_contextflow.build(api)[0]
    model = dataclasses.replace(pi05.model, pi05=False, discrete_state_input=False, max_token_len=48)
    return [
        dataclasses.replace(
            pi05,
            name="ContextFlow_pi0_full",
            assets_repo_override="ContextFlow_pi0_full",
            model=model,
            weight_loader=FullBackboneWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            freeze_filter=model.get_freeze_filter(),
            data=dataclasses.replace(
                pi05.data,
                base_config=dataclasses.replace(pi05.data.base_config, use_quantile_norm=False),
            ),
            policy_metadata={"backbone": "pi0_base", "context": "ContextFlow", "full_pretrained_vlm": True},
        )
    ]
