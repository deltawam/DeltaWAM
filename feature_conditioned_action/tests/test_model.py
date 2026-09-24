"""CPU tests that do not require model checkpoints."""

from dataclasses import replace

import torch
import torch.nn as nn

from starwam.modules.action_dit import ActionDiT

from feature_conditioned_action import (
    DeltaWorldFeatureActionModel,
    DeltaWorldTokenRollout,
)


class _FakePredictor(nn.Module):
    def __init__(self, query_dim=8, world_dim=6):
        super().__init__()
        self.query_projection = nn.Linear(query_dim, world_dim)
        self.last_batch_size = None

    def forward(self, query, kv, mask, pos_q, pos_k, film_condition=None):
        self.last_batch_size = query.shape[0]
        del mask, pos_k, film_condition
        time = pos_q[0].unsqueeze(-1).to(kv.dtype)
        return kv.mean(dim=1, keepdim=True) + self.query_projection(query) + time


class _FakeVFM(nn.Module):
    def __init__(self, world_dim=6):
        super().__init__()
        self.hidden_size = world_dim
        self.projection = nn.Linear(1, world_dim)
        self.calls = 0
        self.inputs = []

    def forward(self, frames):
        self.calls += 1
        self.inputs.append(frames.detach().clone())
        pooled = frames.float().mean(dim=(2, 3, 4)).unsqueeze(-1)
        features = self.projection(pooled)
        return features.unsqueeze(2).expand(-1, -1, 2, -1)


class _FakeTokenizer(nn.Module):
    def __init__(self, world_dim=6):
        super().__init__()
        self.backbone = _FakeVFM(world_dim)
        self.tokenize_calls = 0
        self.decode_calls = 0
        self.decode_batch_sizes = []

    def _rope(self, frames):
        del frames
        return torch.empty(0), torch.empty(0)

    def tokenize_offline(self, initial_feature, vfm_features, rope):
        del rope
        self.tokenize_calls += 1
        previous = torch.cat(
            [initial_feature.unsqueeze(1), vfm_features[:, :-1]], dim=1
        )
        return (vfm_features - previous).mean(dim=2, keepdim=True)

    def decode(self, token, reference, rope):
        del rope
        self.decode_calls += 1
        self.decode_batch_sizes.append(token.shape[0])
        return reference + token


class _FakeDeltaWorld(nn.Module):
    def __init__(self, query_dim=6, world_dim=6):
        super().__init__()
        self.use_bom = True
        self.predictor_hidden_size = query_dim
        self.initializer_range = 0.02
        self.tokenizer = _FakeTokenizer(world_dim)
        self.backbone = self.tokenizer.backbone
        self.predictor = _FakePredictor(query_dim, world_dim)


def _build_model(
    action_history_steps=0,
    action_history_dropout=0.2,
    action_history_attention_mode="separate_gated",
    action_history_gate_init=0.0,
    world_condition_mode="delta_tokens",
):
    world = _FakeDeltaWorld()
    rollout = DeltaWorldTokenRollout(
        world, num_samples=2, freeze_world=True
    )
    action_dit = ActionDiT(
        hidden_dim=16,
        action_dim=4,
        ffn_dim=32,
        text_dim=12,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=8,
        num_layers=2,
        max_seq_len=16,
    )
    return DeltaWorldFeatureActionModel(
        rollout,
        action_dit,
        context_frames=2,
        world_prediction_steps=3,
        world_condition_mode=world_condition_mode,
        include_text=True,
        action_history_steps=action_history_steps,
        action_history_dropout=action_history_dropout,
        action_history_attention_mode=action_history_attention_mode,
        action_history_gate_init=action_history_gate_init,
    )


def test_condition_cache_and_action_forward_shapes():
    model = _build_model()
    model.train()
    assert model.training
    assert not model.world_rollout.training
    assert not model.world_rollout.delta_world.training
    assert not model.world_rollout.delta_world.tokenizer.training
    frames = torch.randint(0, 256, (2, 2, 3, 8, 8), dtype=torch.uint8)
    encoding = model.world_rollout.encode_observations(frames)
    assert encoding.vfm_features.shape == (2, 2, 2, 6)
    assert encoding.initial_vfm_feature.shape == (2, 2, 6)
    assert encoding.delta_tokens.shape == (2, 2, 6)
    assert model.world_rollout.delta_world.tokenizer.backbone.calls == 2
    assert model.world_rollout.delta_world.tokenizer.backbone.inputs[1].dtype == torch.uint8
    assert not model.world_rollout.delta_world.tokenizer.backbone.inputs[1].any()
    assert model.world_rollout.delta_world.tokenizer.tokenize_calls == 1
    text = torch.randn(2, 5, 12)
    condition = model.encode_condition(frames, text_context=text)

    # The reusable cache contains one coherent 3-step future, 2 spatial anchors,
    # and all 5 native text/proprio condition tokens.
    assert condition.future_tokens.shape == (2, 3, 6)
    assert condition.current_vfm_features.shape == (2, 2, 6)
    assert condition.kv_cache.sequence_length == 10
    assert condition.kv_cache.num_anchor_tokens == 2
    assert condition.kv_cache.num_future_tokens == 3
    assert condition.kv_cache.num_native_tokens == 5
    assert condition.kv_cache.key_mask.all()
    assert condition.text_context is text
    assert condition.selected_branch.tolist() == [0, 0]
    assert len(condition.kv_cache.layers) == 2

    action = torch.randn(2, 4, 4)
    output = model(action, torch.tensor([100.0, 200.0]), condition)
    assert output.shape == action.shape

    # Training with future RGB runs both BoM branches, then gathers one oracle trajectory.
    oracle_model = _build_model()
    future_frames = torch.randint(0, 256, (2, 3, 3, 8, 8), dtype=torch.uint8)
    oracle = oracle_model.encode_condition(
        frames, text_context=text, future_frames=future_frames
    )
    assert oracle.future_tokens.shape == (2, 3, 6)
    assert oracle.selected_branch.min() >= 0
    assert oracle.selected_branch.max() < 2
    assert oracle_model.world_rollout.delta_world.predictor.last_batch_size == 4


def test_anchor_only_skips_predictor_and_keeps_current_dino_anchor():
    model = _build_model(world_condition_mode="anchor_only").eval()
    frames = torch.randint(0, 256, (2, 2, 3, 8, 8), dtype=torch.uint8)
    text = torch.randn(2, 5, 12)
    condition = model.encode_condition(frames, text_context=text)

    assert model.world_rollout.delta_world.predictor.last_batch_size is None
    assert condition.future_tokens.shape == (2, 0, 6)
    assert condition.future_mask.shape == (2, 0)
    assert condition.current_vfm_features.shape == (2, 2, 6)
    assert condition.kv_cache.num_anchor_tokens == 2
    assert condition.kv_cache.num_future_tokens == 0
    assert condition.kv_cache.num_native_tokens == 5
    assert condition.kv_cache.sequence_length == 7


def test_decoded_vfm_is_autoregressive_camera_isolated_and_scheme_a():
    model = _build_model(world_condition_mode="decoded_vfm").eval()
    frames = torch.randint(0, 256, (2, 2, 2, 3, 8, 8), dtype=torch.uint8)
    future = torch.randint(0, 256, (2, 3, 2, 3, 8, 8), dtype=torch.uint8)
    condition = model.encode_condition(frames, future_frames=future)

    # [B,L,V,N,D], with one genuine DeltaTok decode per future time step.
    assert condition.future_tokens.shape == (2, 3, 2, 2, 6)
    tokenizer = model.world_rollout.delta_world.tokenizer
    assert tokenizer.decode_calls == 3
    assert tokenizer.decode_batch_sizes == [4, 4, 4]
    assert condition.kv_cache.num_anchor_tokens == 4
    assert condition.kv_cache.num_future_tokens == 12
    assert condition.kv_cache.sequence_length == 16
    assert condition.kv_cache.key_mask.all()

    # Flattening is strictly [t1_v0_patches, t1_v1_patches, t2_v0_patches, ...].
    adapter = model.kv_adapter
    adapter.use_camera_embedding = False
    features = torch.arange(1 * 2 * 2 * 2 * 6, dtype=torch.float32).reshape(
        1, 2, 2, 2, 6
    )
    mask = torch.tensor([[[True, False], [False, True]]])
    flat, flat_mask, future_len, repeats = adapter._prepare_decoded_features(
        features, mask
    )
    assert torch.equal(flat, features.reshape(1, 8, 6))
    assert future_len == 2
    assert repeats == 4
    assert flat_mask.tolist() == [[True, True, False, False, False, False, True, True]]


def test_unified_native_kv_cache_matches_legacy_path_and_is_reused():
    model = _build_model().eval()
    frames = torch.randint(0, 256, (2, 2, 3, 8, 8), dtype=torch.uint8)
    observed_times, future_times = model._default_times(frames)
    world = model.world_rollout(frames, observed_times, future_times)
    text = torch.randn(2, 5, 12)
    text_mask = torch.tensor(
        [[True, True, True, False, True], [True, True, False, False, True]]
    )

    world_only_cache = model.kv_adapter(
        model.action_dit,
        world.future_tokens,
        world.current_vfm_features,
        world.future_mask,
    )
    action = torch.randn(2, 4, 4)
    timestep = torch.tensor([100.0, 200.0])
    legacy = model.action_model(
        action, timestep, world_only_cache, text_context=text, text_mask=text_mask
    )

    calls = {"text_embedding": 0, "key": 0, "value": 0}

    def count(name):
        def hook(_module, _inputs, _output):
            calls[name] += 1
        return hook

    handles = [model.action_dit.text_embedding.register_forward_hook(count("text_embedding"))]
    for block in model.action_dit.blocks:
        handles.append(block.cross_attn.k.register_forward_hook(count("key")))
        handles.append(block.cross_attn.v.register_forward_hook(count("value")))

    unified_cache = model.kv_adapter(
        model.action_dit,
        world.future_tokens,
        world.current_vfm_features,
        world.future_mask,
        text_context=text,
        text_mask=text_mask,
    )
    calls_after_build = calls.copy()
    cached_outputs = [model.action_model(action, timestep, unified_cache) for _ in range(3)]
    for handle in handles:
        handle.remove()

    assert calls_after_build == {"text_embedding": 1, "key": 2, "value": 2}
    assert calls == calls_after_build
    assert unified_cache.num_native_tokens == 5
    assert torch.equal(unified_cache.key_mask[:, -5:], text_mask)
    for cached in cached_outputs:
        assert torch.allclose(cached, legacy, atol=1e-6, rtol=1e-5)


def test_executed_action_history_mask_order_gate_and_gradient():
    model = _build_model(action_history_steps=8, action_history_dropout=0.0).eval()
    frames = torch.randint(0, 256, (2, 2, 3, 8, 8), dtype=torch.uint8)
    text = torch.randn(2, 5, 12)
    history = torch.randn(2, 8, 4)
    history_mask = torch.tensor([
        [False, False, False, False, False, True, True, True],
        [False, False, True, True, True, True, True, True],
    ])
    condition = model.encode_condition(
        frames, text_context=text, action_history=history,
        action_history_mask=history_mask,
    )

    # Primary cache is 2 anchor + 3 future + 5 text. History is a separate
    # 8-token memory with an independent softmax and mask.
    cache = condition.kv_cache
    assert cache.sequence_length == 10
    assert cache.total_sequence_length == 18
    assert cache.num_anchor_tokens == 2
    assert cache.num_future_tokens == 3
    assert cache.num_native_tokens == 5
    assert cache.num_action_history_tokens == 8
    assert cache.key_mask.shape == (2, 10)
    assert cache.history_layers is not None
    assert len(cache.history_layers) == 2
    assert cache.history_layers[0].key.shape[2] == 8
    assert torch.equal(cache.history_key_mask, history_mask)
    assert torch.equal(condition.action_history_mask, history_mask)
    assert cache.attention_bias is None
    assert torch.equal(model.action_model.history_gate_values(), torch.zeros(2))

    action = torch.randn(2, 4, 4)
    timestep = torch.tensor([100.0, 200.0])
    output = model(action, timestep, condition)
    primary_only_cache = replace(
        cache, history_layers=None, history_key_mask=None,
        num_action_history_tokens=0,
    )
    primary_only = model.action_model(action, timestep, primary_only_cache)
    # Zero-init residual gate makes the old checkpoint path exactly identical.
    assert torch.equal(output, primary_only)
    output.sum().backward()
    assert model.action_model.history_gates.grad is not None
    assert model.action_model.history_gates.grad.abs().sum() > 0


def test_unified_action_history_mode_remains_available_for_ablation():
    model = _build_model(
        action_history_steps=8, action_history_dropout=0.0,
        action_history_attention_mode="unified", action_history_gate_init=0.1,
    ).eval()
    frames = torch.randint(0, 256, (2, 2, 3, 8, 8), dtype=torch.uint8)
    history = torch.randn(2, 8, 4)
    mask = torch.ones(2, 8, dtype=torch.bool)
    condition = model.encode_condition(
        frames, action_history=history, action_history_mask=mask
    )
    cache = condition.kv_cache
    assert cache.sequence_length == 13  # 2 anchor + 3 future + 8 history
    assert cache.total_sequence_length == 13
    assert cache.history_layers is None
    assert cache.attention_bias is not None
    assert torch.allclose(model.action_model.history_gate_values(), torch.tensor([0.1]))


def test_action_history_dropout_masks_whole_samples():
    model = _build_model(action_history_steps=8, action_history_dropout=1.0).train()
    actions = torch.randn(3, 8, 4)
    hidden, mask = model.action_history_encoder(
        actions, torch.ones(3, 8, dtype=torch.bool)
    )
    assert hidden.shape == (3, 8, 16)
    assert not mask.any()


def test_all_masked_separate_history_is_exact_noop_with_nonzero_gate():
    model = _build_model(
        action_history_steps=8,
        action_history_dropout=0.0,
        action_history_attention_mode="separate_gated",
        action_history_gate_init=0.1,
    ).eval()
    frames = torch.randint(0, 256, (2, 2, 3, 8, 8), dtype=torch.uint8)
    history = torch.randn(2, 8, 4)
    condition = model.encode_condition(
        frames,
        action_history=history,
        action_history_mask=torch.zeros(2, 8, dtype=torch.bool),
    )
    action = torch.randn(2, 4, 4)
    timestep = torch.tensor([100.0, 200.0])
    with_history = model(action, timestep, condition)
    primary_only_cache = replace(
        condition.kv_cache,
        history_layers=None,
        history_key_mask=None,
        num_action_history_tokens=0,
    )
    primary_only = model.action_model(action, timestep, primary_only_cache)
    assert torch.equal(with_history, primary_only)


def test_oracle_branch_probability_controls_training_selection():
    model = _build_model()
    oracle = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    model.world_rollout.oracle_branch_probability = 1.0
    assert model.world_rollout._select_training_branch(oracle, 2, None).tolist() == [1, 1, 1, 1]

    generator = torch.Generator().manual_seed(5)
    model.world_rollout.oracle_branch_probability = 0.0
    selected = model.world_rollout._select_training_branch(oracle, 2, generator)
    assert selected.shape == oracle.shape
    assert selected.min() >= 0
    assert selected.max() < 2

    model.world_rollout.oracle_branch_probability = 0.5
    generator = torch.Generator().manual_seed(7)
    selected = model.world_rollout._select_training_branch(oracle, 2, generator)
    assert selected.shape == oracle.shape
    assert selected.min() >= 0
    assert selected.max() < 2


def test_training_step_and_cached_sampling():
    model = _build_model()
    sample = {
        "video": torch.randint(0, 256, (2, 3, 4, 8, 8), dtype=torch.uint8),
        "action": torch.randn(2, 4, 4),
        "context": torch.randn(2, 5, 12),
        "context_mask": torch.ones(2, 5, dtype=torch.bool),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }
    loss, metrics = model.training_step(sample)
    loss.backward()
    assert loss.ndim == 0
    assert metrics["condition_tokens"] == 10.0

    frames = sample["video"].permute(0, 2, 1, 3, 4)[:, :2]
    actions = model.sample_actions(
        frames,
        action_horizon=4,
        num_inference_steps=2,
        text_context=sample["context"],
        text_mask=sample["context_mask"],
        seed=7,
    )
    assert actions.shape == (2, 4, 4)


def test_rejects_normalized_rgb_and_misaligned_timestamps():
    model = _build_model()
    raw = torch.randint(0, 256, (2, 2, 3, 8, 8), dtype=torch.uint8)
    try:
        model.world_rollout.encode_observations(raw.float().div(255))
    except TypeError as error:
        assert "raw uint8 RGB" in str(error)
    else:
        raise AssertionError("pre-normalized RGB must be rejected")

    observed_times = torch.tensor([[0.0, 0.1], [0.0, 0.1]])
    invalid_future_times = torch.tensor([[0.1, 0.2], [0.2, 0.3]])
    try:
        model.world_rollout(raw, observed_times, invalid_future_times)
    except ValueError as error:
        assert "after the current frame" in str(error)
    else:
        raise AssertionError("future timestamps overlapping context must be rejected")


def test_optional_cross_attention_diagnostics_are_normalized():
    model = _build_model().eval()
    model.kv_adapter.capture_condition_hidden = True
    frames = torch.randint(0, 256, (2, 2, 3, 8, 8), dtype=torch.uint8)
    text = torch.randn(2, 5, 12)
    condition = model.encode_condition(frames, text_context=text)
    assert condition.kv_cache.condition_hidden is not None
    assert condition.kv_cache.condition_hidden.shape == (2, 10, 16)

    action = torch.randn(2, 4, 4)
    output, attention = model.action_model(
        action,
        torch.tensor([100.0, 200.0]),
        condition.kv_cache,
        return_cross_attention=True,
        cross_attention_layers=(1,),
    )
    assert output.shape == action.shape
    assert set(attention) == {1}
    assert attention[1].shape == (2, 2, 4, 10)
    expected = torch.ones(2, 2, 4)
    assert torch.allclose(attention[1].sum(dim=-1), expected, atol=1e-5)
