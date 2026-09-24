"""Full pretrained backbone experiment, with the original held-out LIBERO split."""

import dataclasses

from openpi.models.full_contextflow import FullContextFlowConfig
from openpi.training import config_libero
from openpi.training.full_backbone_weights import FullBackboneWeightLoader


def build(api):
    original = next(config for config in config_libero.build(api) if config.name == "ContextFlow")
    model = FullContextFlowConfig(
        pi05=True,
        state_dim=8,
        native_action_dim=7,
        sample_frames=original.model.sample_frames,
        sample_actions=original.model.sample_actions,
        random_select=original.model.random_select,
    )
    return [
        dataclasses.replace(
            original,
            name="ContextFlow_pi05_full",
            assets_repo_override="ContextFlow_pi05_full",
            model=model,
            weight_loader=FullBackboneWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            freeze_filter=model.get_freeze_filter(),
            data=dataclasses.replace(
                original.data,
                base_config=dataclasses.replace(original.data.base_config, use_quantile_norm=True),
            ),
            num_workers=8,
            policy_metadata={"backbone": "pi05_base", "context": "ContextFlow", "full_pretrained_vlm": True},
        )
    ]
