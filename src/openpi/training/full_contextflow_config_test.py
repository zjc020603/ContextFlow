import dataclasses

import numpy as np
import pytest

from openpi import transforms
from openpi.models import tokenizer
from openpi.training import config
from openpi.training.full_backbone_weights import FullBackboneWeightLoader


def test_pi05_full_preserves_split_and_context_settings():
    old = config.get_config("ContextFlow")
    new = config.get_config("ContextFlow_pi05_full")
    assert new.model.paligemma_variant == "gemma_2b"
    assert new.model.action_expert_variant == "gemma_300m"
    assert new.model.pi05
    assert new.model.discrete_state_input
    assert new.model.max_token_len == 200
    assert new.model.state_dim == 8
    assert new.model.native_action_dim == 7
    assert new.model.action_horizon == old.model.action_horizon
    assert new.data.remove_task_list == old.data.remove_task_list
    assert new.data.episode_json_path == old.data.episode_json_path
    assert new.data.sample_frames == old.data.sample_frames == new.model.sample_frames
    assert new.data.sample_actions == old.data.sample_actions == new.model.sample_actions
    assert new.data.norm_stats_aliases == old.data.norm_stats_aliases
    assert new.data.base_config.use_quantile_norm
    assert new.use_custom_dataloader
    assert not new.data.use_delta_joint_actions
    assert isinstance(new.weight_loader, FullBackboneWeightLoader)
    assert new.weight_loader.params_path.endswith("/pi05_base/params")
    assert new.assets_dirs != old.assets_dirs


def test_discrete_state_format_and_padding_match_native_input():
    tok = tokenizer.PaligemmaTokenizer(200)
    state = np.linspace(-1, 1, 8, dtype=np.float32)
    padded = np.pad(state, (0, 24), constant_values=-1)
    actions = np.ones((3, 32))
    data = {
        "prompt": "pick_up\nthe cup",
        "state": padded,
        "actions": actions,
        "dem_prompt_all_states": np.tile(padded, (5, 1)),
        "dem_prompt_all_actions": np.ones((5, 32)),
    }
    restored = transforms.RestoreNativePadding(8, 7)(data)
    result = transforms.TokenizePrompt(tok, discrete_state_input=True, state_dim=8)(restored)
    expected, mask = tok.tokenize("pick_up\nthe cup", state)
    np.testing.assert_array_equal(result["tokenized_prompt"], expected)
    np.testing.assert_array_equal(result["tokenized_prompt_mask"], mask)
    np.testing.assert_array_equal(result["state"][:8], state)
    for key, dim in [("state", 8), ("actions", 7), ("dem_prompt_all_states", 8), ("dem_prompt_all_actions", 7)]:
        np.testing.assert_array_equal(result[key][..., dim:], 0)
    # Transform does not mutate its caller's tensors.
    np.testing.assert_array_equal(data["state"][8:], -1)
    np.testing.assert_array_equal(actions, 1)


def test_invalid_full_backbone_settings_fail_early():
    c = config.get_config("ContextFlow_pi05_full").model
    with pytest.raises(ValueError, match="complete pretrained"):
        dataclasses.replace(c, paligemma_variant="gemma_300m_v2")
    with pytest.raises(ValueError, match="language/state"):
        dataclasses.replace(c, use_text_prompts=False)
    with pytest.raises(ValueError, match="unpooled"):
        dataclasses.replace(c, avg_current_img=True)


def test_full_config_exposes_local_training_data_switch(monkeypatch):
    c = config.get_config("ContextFlow_pi05_full")
    monkeypatch.setattr(config.DataConfigFactory, "create_base_config", lambda self, assets: self.base_config)
    local = dataclasses.replace(c.data, local_files_only=True)
    assert local.create_base_config(c.assets_dirs).local_files_only
    assert not c.data.create_base_config(c.assets_dirs).local_files_only
