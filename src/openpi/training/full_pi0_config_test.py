from openpi.training import config


def test_full_pi0_is_a_matched_control():
    pi0 = config.get_config("ContextFlow_pi0_full")
    pi05 = config.get_config("ContextFlow_pi05_full")
    assert not pi0.model.pi05
    assert not pi0.model.discrete_state_input
    assert pi0.model.max_token_len == 48
    assert pi0.model.paligemma_variant == pi05.model.paligemma_variant == "gemma_2b"
    assert pi0.model.action_expert_variant == pi05.model.action_expert_variant == "gemma_300m"
    assert pi0.model.action_horizon == pi05.model.action_horizon
    assert pi0.data.remove_task_list == pi05.data.remove_task_list
    assert pi0.data.sample_frames == pi05.data.sample_frames
    assert pi0.data.sample_actions == pi05.data.sample_actions
    assert pi0.batch_size == pi05.batch_size
    assert pi0.seed == pi05.seed
    assert pi0.num_train_steps == pi05.num_train_steps
    assert pi0.optimizer == pi05.optimizer
    assert pi0.lr_schedule == pi05.lr_schedule
    assert pi0.data.norm_stats_aliases == pi05.data.norm_stats_aliases
    assert not pi0.data.base_config.use_quantile_norm
    assert pi0.weight_loader.params_path.endswith("/pi0_base/params")
    assert pi0.assets_dirs != pi05.assets_dirs
