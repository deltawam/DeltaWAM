import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_conditioned_action.text_conditioning import T5InstructionEncoder
from models.deltatok import DeltaTok
from models.predictor import Predictor
from models.world import World, causal_mask


class SplitPoseGripperActionEncoder(nn.Module):
    """Encode continuous pose commands and gripper state on separate paths.

    Pose normalization uses fixed training-set statistics stored as buffers, so
    training, offline evaluation, and closed-loop rollout share exactly the
    same transform. The scalar gripper command deliberately bypasses pose
    normalization and receives its own MLP before feature fusion.
    """

    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        pose_mean: tuple[float, ...],
        pose_std: tuple[float, ...],
        pose_hidden_size: int = 512,
        gripper_hidden_size: int = 128,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        action_dim = int(action_dim)
        pose_dim = action_dim - 1
        if pose_dim <= 0:
            raise ValueError("split action encoder requires pose dims plus gripper")
        if len(pose_mean) != pose_dim or len(pose_std) != pose_dim:
            raise ValueError(
                "pose action statistics must match action_dim - 1: "
                f"expected {pose_dim}, got mean={len(pose_mean)}, std={len(pose_std)}"
            )
        mean = torch.as_tensor(pose_mean, dtype=torch.float32)
        std = torch.as_tensor(pose_std, dtype=torch.float32)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("pose action statistics must be finite")
        if not bool((std > 0).all()):
            raise ValueError("every pose action std must be positive")

        self.action_dim = action_dim
        self.pose_dim = pose_dim
        self.register_buffer("pose_mean", mean)
        self.register_buffer("pose_std", std)
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, int(pose_hidden_size)),
            nn.SiLU(),
            nn.Linear(int(pose_hidden_size), int(pose_hidden_size)),
        )
        self.gripper_encoder = nn.Sequential(
            nn.Linear(1, int(gripper_hidden_size)),
            nn.SiLU(),
            nn.Linear(int(gripper_hidden_size), int(gripper_hidden_size)),
        )
        self.projection = nn.Linear(
            int(pose_hidden_size) + int(gripper_hidden_size), int(hidden_size)
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=float(initializer_range))
                nn.init.zeros_(module.bias)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected action last dim {self.action_dim}, got {action.shape[-1]}"
            )
        pose = action[..., : self.pose_dim].float()
        gripper = action[..., self.pose_dim :].float()
        pose = (pose - self.pose_mean) / self.pose_std
        pose_embedding = self.pose_encoder(pose)
        gripper_embedding = self.gripper_encoder(gripper)
        return self.projection(
            torch.cat([pose_embedding, gripper_embedding], dim=-1)
        )


class SensorNoiseAugmenter(nn.Module):
    """Apply camera-realistic corruption before VFM normalization.

    Multi-view parameters are sampled independently per physical camera. Blur
    and occlusion remain fixed over the temporal window for a given camera,
    while Gaussian shot noise is sampled independently for every pixel/frame.
    """

    def __init__(
        self,
        enabled: bool = False,
        probability: float = 0.75,
        clean_probability: float | None = None,
        gaussian_probability: float = 0.5,
        gaussian_std_range: tuple[float, float] = (0.0, 8.0),
        occlusion_probability: float = 0.2,
        occlusion_area_range: tuple[float, float] = (0.02, 0.1),
        motion_blur_probability: float = 0.25,
        motion_blur_kernel_sizes: tuple[int, ...] = (3, 5),
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.probability = self._probability(probability, "probability")
        self.clean_probability = (
            1.0 - self.probability
            if clean_probability is None
            else self._probability(clean_probability, "clean_probability")
        )
        if abs(self.probability + self.clean_probability - 1.0) > 1e-8:
            raise ValueError(
                "sensor noise and clean probabilities must sum to 1"
            )
        self.gaussian_probability = self._probability(
            gaussian_probability, "gaussian_probability"
        )
        self.occlusion_probability = self._probability(
            occlusion_probability, "occlusion_probability"
        )
        self.motion_blur_probability = self._probability(
            motion_blur_probability, "motion_blur_probability"
        )
        self.gaussian_std_range = self._ordered_range(
            gaussian_std_range, "gaussian_std_range", minimum=0.0
        )
        self.occlusion_area_range = self._ordered_range(
            occlusion_area_range, "occlusion_area_range", minimum=0.0, maximum=1.0
        )
        kernels = tuple(int(size) for size in motion_blur_kernel_sizes)
        if not kernels or any(size < 3 or size % 2 == 0 for size in kernels):
            raise ValueError(
                "motion_blur_kernel_sizes must contain positive odd sizes >= 3"
            )
        self.motion_blur_kernel_sizes = kernels
        self.last_stats: dict[str, torch.Tensor] = {}

    @staticmethod
    def _probability(value: float, name: str) -> float:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0,1], got {value}")
        return value

    @staticmethod
    def _ordered_range(
        value: tuple[float, float],
        name: str,
        minimum: float,
        maximum: float | None = None,
    ) -> tuple[float, float]:
        low, high = map(float, value)
        if low < minimum or high < low or (maximum is not None and high > maximum):
            raise ValueError(f"invalid {name}: {(low, high)}")
        return low, high

    @staticmethod
    def _motion_kernel(
        size: int, direction: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        kernel = torch.zeros((size, size), device=device, dtype=dtype)
        if direction == 0:
            kernel[size // 2] = 1
        elif direction == 1:
            kernel[:, size // 2] = 1
        elif direction == 2:
            kernel.fill_diagonal_(1)
        else:
            kernel = torch.flip(torch.eye(size, device=device, dtype=dtype), (1,))
        return kernel / kernel.sum()

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            self.last_stats = {}
            return frames
        if frames.ndim not in (5, 6):
            raise ValueError(
                "sensor noise expects [B,T,C,H,W] or [B,T,V,C,H,W], "
                f"got {tuple(frames.shape)}"
            )

        single_view = frames.ndim == 5
        source = frames.unsqueeze(2) if single_view else frames
        out = source.float().clone()
        batch, _, views, channels, height, width = out.shape
        active = torch.rand(batch, device=out.device) < self.probability
        gaussian_count = occlusion_count = blur_count = 0

        for batch_index in range(batch):
            if not bool(active[batch_index]):
                continue
            for view_index in range(views):
                view = out[batch_index, :, view_index]
                if bool(torch.rand((), device=out.device) < self.gaussian_probability):
                    low, high = self.gaussian_std_range
                    std = low + (high - low) * torch.rand((), device=out.device)
                    view.add_(torch.randn_like(view) * std)
                    gaussian_count += 1

                if bool(torch.rand((), device=out.device) < self.occlusion_probability):
                    low, high = self.occlusion_area_range
                    area_fraction = low + (high - low) * torch.rand((), device=out.device)
                    aspect = torch.exp(
                        (torch.rand((), device=out.device) * 2.0 - 1.0) * 0.5
                    )
                    block_h = max(
                        1,
                        min(height, int(round(float(torch.sqrt(area_fraction / aspect)) * height))),
                    )
                    block_w = max(
                        1,
                        min(width, int(round(float(torch.sqrt(area_fraction * aspect)) * width))),
                    )
                    top = int(torch.randint(height - block_h + 1, (), device=out.device))
                    left = int(torch.randint(width - block_w + 1, (), device=out.device))
                    view[:, :, top : top + block_h, left : left + block_w] = 0.0
                    occlusion_count += 1

                if bool(torch.rand((), device=out.device) < self.motion_blur_probability):
                    kernel_index = int(
                        torch.randint(len(self.motion_blur_kernel_sizes), (), device=out.device)
                    )
                    size = self.motion_blur_kernel_sizes[kernel_index]
                    direction = int(torch.randint(4, (), device=out.device))
                    kernel = self._motion_kernel(size, direction, out.device, out.dtype)
                    weight = kernel.view(1, 1, size, size).expand(channels, 1, -1, -1)
                    pad = size // 2
                    padded = F.pad(view, (pad, pad, pad, pad), mode="reflect")
                    view.copy_(F.conv2d(padded, weight, groups=channels))
                    blur_count += 1

        total_views = max(batch * views, 1)
        self.last_stats = {
            "active_sample_fraction": active.float().mean(),
            "gaussian_view_fraction": out.new_tensor(gaussian_count / total_views),
            "occlusion_view_fraction": out.new_tensor(occlusion_count / total_views),
            "motion_blur_view_fraction": out.new_tensor(blur_count / total_views),
        }
        out = out.clamp_(0.0, 255.0)
        if not frames.is_floating_point():
            out = out.round().to(frames.dtype)
        else:
            out = out.to(frames.dtype)
        return out[:, :, 0] if single_view else out


class DeltaWorld(World):
    def __init__(
        self,
        tokenizer: DeltaTok,
        rope_axis_sizes: tuple = (60,),
        predictor_hidden_size: int = 768,
        predictor_num_hidden_layers: int = 12,
        predictor_num_heads: int = 12,
        use_bom: bool = True,
        num_samples_train: int = 256,
        num_samples_eval: int = 20,
        layer_scale_init: float = 1e-5,
        rope_unrotated_size: int = 4,
        mlp_ratio: int = 4,
        max_cameras: int = 4,
        use_camera_embedding: bool = False,
        use_task_film: bool = False,
        task_film_text_encoder_name: str = "google-t5/t5-base",
        task_film_text_encoder_max_length: int = 64,
        freeze_task_film_text_encoder: bool = True,
        task_film_scale: float = 0.1,
        task_film_gated_residual: bool = False,
        use_task_cross_attn: bool = False,
        task_cross_attn_gate_init: float = 0.0,
        task_cross_attn_scale: float = 1.0,
        use_action_conditioning: bool = False,
        action_dim: int = 7,
        action_encoder_type: str = "legacy_layernorm_mlp",
        action_pose_mean: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        action_pose_std: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        action_pose_hidden_size: int = 512,
        action_gripper_hidden_size: int = 128,
        action_film_scale: float = 1.0,
        action_film_gated_residual: bool = True,
        sensor_noise_enabled: bool = False,
        sensor_noise_probability: float = 0.75,
        sensor_clean_probability: float | None = None,
        sensor_gaussian_probability: float = 0.5,
        sensor_gaussian_std_range: tuple[float, float] = (0.0, 8.0),
        sensor_occlusion_probability: float = 0.2,
        sensor_occlusion_area_range: tuple[float, float] = (0.02, 0.1),
        sensor_motion_blur_probability: float = 0.25,
        sensor_motion_blur_kernel_sizes: tuple[int, ...] = (3, 5),
    ):
        super().__init__(
            **{k: v for k, v in locals().items() if k not in ("self", "__class__")},
            initializer_range=tokenizer.backbone.initializer_range,
        )
        tokenizer.requires_grad_(False).eval()
        self.backbone = tokenizer.backbone
        self._validate_rope_sizes()

        self.predictor = Predictor(
            self.backbone.hidden_size,
            self.backbone.initializer_range,
            rope_axis_sizes,
            rope_unrotated_size,
            layer_scale_init,
            predictor_hidden_size,
            predictor_num_hidden_layers,
            predictor_num_heads,
            mlp_ratio,
            film_condition_dim=predictor_hidden_size if use_task_film else None,
            film_scale=task_film_scale,
            film_gated_residual=task_film_gated_residual,
            use_task_cross_attn=use_task_cross_attn,
            task_cross_attn_gate_init=task_cross_attn_gate_init,
            task_cross_attn_scale=task_cross_attn_scale,
            action_film_condition_dim=(
                predictor_hidden_size if use_action_conditioning else None
            ),
            action_film_scale=action_film_scale,
            action_film_gated_residual=action_film_gated_residual,
        )
        self.use_camera_embedding = bool(use_camera_embedding)
        self.camera_embedding = nn.Embedding(max_cameras, self.backbone.hidden_size)
        nn.init.normal_(self.camera_embedding.weight, std=0.02)
        # Single-view legacy configs leave this frozen/unused. Multi-view configs
        # set use_camera_embedding=True, making the new view tags trainable and
        # optimizer-visible even when warm-starting from single-camera checkpoints.
        self.camera_embedding.requires_grad_(self.use_camera_embedding)
        self.use_task_film = bool(use_task_film)
        self.use_task_cross_attn = bool(use_task_cross_attn)
        self.use_action_conditioning = bool(use_action_conditioning)
        self.action_dim = int(action_dim)
        action_encoder_type = str(action_encoder_type).strip().lower()
        if action_encoder_type not in {"legacy_layernorm_mlp", "split_pose_gripper"}:
            raise ValueError(
                "action_encoder_type must be 'legacy_layernorm_mlp' or "
                f"'split_pose_gripper', got {action_encoder_type!r}"
            )
        self.action_encoder_type = action_encoder_type
        if not self.use_action_conditioning:
            self.action_encoder = None
        elif action_encoder_type == "legacy_layernorm_mlp":
            self.action_encoder = nn.Sequential(
                nn.LayerNorm(self.action_dim),
                nn.Linear(self.action_dim, predictor_hidden_size),
                nn.SiLU(),
                nn.Linear(predictor_hidden_size, predictor_hidden_size),
            )
        else:
            self.action_encoder = SplitPoseGripperActionEncoder(
                action_dim=self.action_dim,
                hidden_size=predictor_hidden_size,
                pose_mean=tuple(action_pose_mean),
                pose_std=tuple(action_pose_std),
                pose_hidden_size=action_pose_hidden_size,
                gripper_hidden_size=action_gripper_hidden_size,
                initializer_range=self.backbone.initializer_range,
            )
        if (
            self.action_encoder is not None
            and action_encoder_type == "legacy_layernorm_mlp"
        ):
            for module in self.action_encoder.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(
                        module.weight, std=self.backbone.initializer_range
                    )
                    nn.init.zeros_(module.bias)
        self.task_film_encoder = (
            T5InstructionEncoder(
                model_name_or_path=task_film_text_encoder_name,
                output_dim=predictor_hidden_size,
                max_length=task_film_text_encoder_max_length,
                freeze_encoder=freeze_task_film_text_encoder,
            )
            if (self.use_task_film or self.use_task_cross_attn)
            else None
        )
        self.sensor_noise = SensorNoiseAugmenter(
            enabled=sensor_noise_enabled,
            probability=sensor_noise_probability,
            clean_probability=sensor_clean_probability,
            gaussian_probability=sensor_gaussian_probability,
            gaussian_std_range=sensor_gaussian_std_range,
            occlusion_probability=sensor_occlusion_probability,
            occlusion_area_range=sensor_occlusion_area_range,
            motion_blur_probability=sensor_motion_blur_probability,
            motion_blur_kernel_sizes=sensor_motion_blur_kernel_sizes,
        )

    def _encode_task_condition(
        self, condition
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not (self.use_task_film or self.use_task_cross_attn):
            return None, None, None
        if condition is None:
            raise ValueError(
                "DeltaWorld task conditioning requires an instruction condition when "
                "use_task_film=True or use_task_cross_attn=True"
            )
        if torch.is_tensor(condition):
            if condition.ndim == 2:
                if condition.shape[-1] != self.predictor_hidden_size:
                    raise ValueError(
                        "tensor task condition must have last dim predictor_hidden_size, "
                        f"got {tuple(condition.shape)}"
                    )
                tokens = condition[:, None]
                mask = torch.ones(
                    tokens.shape[:2], device=tokens.device, dtype=torch.bool
                )
                pooled = condition
            elif condition.ndim == 3:
                if condition.shape[-1] != self.predictor_hidden_size:
                    raise ValueError(
                        "tensor task memory must have last dim predictor_hidden_size, "
                        f"got {tuple(condition.shape)}"
                    )
                tokens = condition
                mask = torch.ones(
                    tokens.shape[:2], device=tokens.device, dtype=torch.bool
                )
                pooled = tokens.mean(dim=1)
            else:
                raise ValueError(
                    "tensor task condition must be [B,D] or [B,L,D], "
                    f"got {tuple(condition.shape)}"
                )
        else:
            tokens, mask = self.task_film_encoder(condition)
            weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
            denom = weights.sum(dim=1).clamp(min=1.0)
            pooled = (tokens * weights).sum(dim=1) / denom

        film_condition = pooled if self.use_task_film else None
        task_memory = tokens if self.use_task_cross_attn else None
        task_memory_mask = mask if self.use_task_cross_attn else None
        return film_condition, task_memory, task_memory_mask

    def _encode_task_film_condition(self, condition) -> torch.Tensor | None:
        film_condition, _, _ = self._encode_task_condition(condition)
        return film_condition

    def _encode_action_condition(
        self,
        action_condition: torch.Tensor | None,
        expected_queries: int,
    ) -> torch.Tensor | None:
        if not self.use_action_conditioning:
            if action_condition is not None:
                raise ValueError(
                    "action_condition was provided but use_action_conditioning=False"
                )
            return None
        if action_condition is None:
            raise ValueError(
                "action-conditioned DeltaWorld requires one action per future query"
            )
        if action_condition.ndim != 3:
            raise ValueError(
                "action_condition must be [B,Q,A], "
                f"got {tuple(action_condition.shape)}"
            )
        if action_condition.shape[1] != expected_queries:
            raise ValueError(
                "action/query temporal mismatch: expected exactly "
                f"{expected_queries} actions, got {action_condition.shape[1]}"
            )
        if action_condition.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected action_dim={self.action_dim}, got {action_condition.shape[-1]}"
            )
        return self.action_encoder(action_condition.float())

    def _camera_ids(self, num_views: int, device: torch.device) -> torch.Tensor:
        if num_views > self.camera_embedding.num_embeddings:
            raise ValueError(
                f"num_views={num_views} exceeds camera embedding capacity "
                f"{self.camera_embedding.num_embeddings}"
            )
        return torch.arange(num_views, device=device)

    def _add_camera_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_camera_embedding:
            return tokens
        num_views = tokens.shape[-2]
        camera = self.camera_embedding(self._camera_ids(num_views, tokens.device)).to(
            device=tokens.device, dtype=tokens.dtype
        )
        return tokens + camera.view(*([1] * (tokens.ndim - 2)), num_views, -1)

    def _forward_train(
        self,
        frames: torch.Tensor,
        timestamps: torch.Tensor,
        criterion: nn.Module | None,
        condition=None,
        action_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(frames, timestamps, criterion)
        film_condition, task_memory, task_memory_mask = self._encode_task_condition(condition)

        # The frozen VFM/DeltaTok encodes a paired view of the same trajectory:
        # corrupted observations supply causal K/V, while clean physical DeltaTok
        # tokens remain the detached supervision and BoM-oracle reference.
        z_clean, _, _ = self._encode_frames(frames)
        if self.sensor_noise.enabled:
            noisy_frames = self.sensor_noise(frames)
            z_input, _, _ = self._encode_frames(noisy_frames)
        else:
            z_input = z_clean
        batch_size, seq_len = z_clean.shape[0], timestamps.shape[1] - 1
        start = 1
        num_queries = timestamps.shape[1] - start
        encoded_actions = self._encode_action_condition(
            action_condition, num_queries
        )

        if z_clean.ndim == 3:
            pos_q = (timestamps[:, start:],)
            pos_k = (timestamps[:, :-1],)
            if self.use_bom:
                q = self._bom_queries(
                    batch_size, num_queries, z_clean.device, z_input[:, :seq_len],
                    pos_q, pos_k, z_clean[:, start:], criterion,
                    film_condition=film_condition,
                    task_memory=task_memory,
                    task_memory_mask=task_memory_mask,
                    action_condition=encoded_actions,
                )
            else:
                q = self._prepare_queries(batch_size, num_queries)
            q_indices = torch.arange(num_queries, device=z_clean.device) + start
            k_indices = torch.arange(seq_len, device=z_clean.device)
            z_hat = self.predictor(
                q, z_input[:, :seq_len], causal_mask(q_indices, k_indices), pos_q, pos_k,
                film_condition, task_memory, task_memory_mask,
                encoded_actions,
            )
            return z_hat, z_clean[:, start:]

        # Multi-view z: [B,T,V,D]. Predictor KV uses scheme A:
        # [t0_cam0, t0_cam1, t1_cam0, t1_cam1, ...]. Camera embeddings tag
        # keys/queries, but the target remains the physical DeltaTok token.
        num_views = z_clean.shape[2]
        z_key = self._add_camera_embedding(z_input)
        z_key = z_key.reshape(batch_size, timestamps.shape[1] * num_views, -1)
        z_tgt = z_clean.reshape(batch_size, timestamps.shape[1] * num_views, -1)
        pos_q = (timestamps[:, start:],)
        pos_k = (timestamps[:, :-1],)
        if self.use_bom:
            q = self._bom_queries(
                batch_size, num_queries, z_clean.device, z_key[:, :seq_len * num_views],
                pos_q, pos_k, z_tgt[:, start * num_views:], criterion,
                tokens_per_frame=num_views,
                film_condition=film_condition,
                task_memory=task_memory,
                task_memory_mask=task_memory_mask,
                action_condition=encoded_actions,
            )
        else:
            q = self._prepare_queries(batch_size, num_queries).repeat_interleave(num_views, 1)
        q = self._add_camera_embedding(q.reshape(batch_size, num_queries, num_views, -1))
        q = q.reshape(batch_size, num_queries * num_views, -1)
        q_indices = torch.arange(num_queries * num_views, device=z_clean.device) // num_views + start
        k_indices = torch.arange(seq_len * num_views, device=z_clean.device) // num_views
        pos_q_flat = (timestamps[:, start:].repeat_interleave(num_views, 1),)
        pos_k_flat = (timestamps[:, :-1].repeat_interleave(num_views, 1),)
        z_hat = self.predictor(
            q, z_key[:, :seq_len * num_views], causal_mask(q_indices, k_indices),
            pos_q_flat, pos_k_flat, film_condition, task_memory, task_memory_mask,
            None if encoded_actions is None else encoded_actions.repeat_interleave(num_views, 1),
        )
        return z_hat, z_tgt[:, start * num_views:]

    def rollout_init(
        self,
        frames: torch.Tensor,
        ctx_len: int,
        condition=None,
        action_condition: torch.Tensor | None = None,
    ) -> dict:
        film_condition, task_memory, task_memory_mask = self._encode_task_condition(condition)
        z, y, rope = self._encode_frames(frames)
        batch_size = y.shape[0]
        rollout_steps = frames.shape[1] - ctx_len
        encoded_actions = self._encode_action_condition(
            action_condition, rollout_steps
        )
        expanded_actions = (
            None
            if encoded_actions is None
            else self._expand_bom(encoded_actions, self._num_samples, batch_size)
        )

        if z.ndim == 3:
            return {
                "kv": self._expand_bom(
                    z[:, :ctx_len].detach(), self._num_samples, batch_size
                ),
                "x": self._expand_bom(y[:, ctx_len - 1], self._num_samples, batch_size),
                "y": y,
                "rope": rope,
                "num_views": 1,
                "ctx_len": ctx_len,
                "action_condition": expanded_actions,
                "film_condition": None if film_condition is None else self._expand_bom(film_condition, self._num_samples, batch_size),
                "task_memory": None if task_memory is None else self._expand_bom(task_memory, self._num_samples, batch_size),
                "task_memory_mask": None if task_memory_mask is None else self._expand_bom(task_memory_mask, self._num_samples, batch_size),
            }

        # Multi-view z: [B,T,V,D]. Eval rollout uses the same scheme-A KV
        # ordering as training: [t0_cam0, t0_cam1, t1_cam0, ...].
        num_views = z.shape[2]
        z_key = self._add_camera_embedding(z)
        kv = z_key[:, :ctx_len].reshape(batch_size, ctx_len * num_views, -1)
        return {
            "kv": self._expand_bom(kv.detach(), self._num_samples, batch_size),
            "x": self._expand_bom(y[:, ctx_len - 1], self._num_samples, batch_size),
            "y": y,
            "rope": rope,
            "num_views": num_views,
            "ctx_len": ctx_len,
            "action_condition": expanded_actions,
            "film_condition": None if film_condition is None else self._expand_bom(film_condition, self._num_samples, batch_size),
            "task_memory": None if task_memory is None else self._expand_bom(task_memory, self._num_samples, batch_size),
            "task_memory_mask": None if task_memory_mask is None else self._expand_bom(task_memory_mask, self._num_samples, batch_size),
        }

    def rollout_step(
        self,
        state: dict,
        tgt_frame_idx: int,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        batch_size = state["y"].shape[0]
        num_views = int(state.get("num_views", 1))
        rollout_step = tgt_frame_idx - int(state["ctx_len"])
        action_condition = state.get("action_condition")
        step_action = (
            None
            if action_condition is None
            else action_condition[:, rollout_step : rollout_step + 1]
        )

        if num_views == 1:
            q = self._rollout_queries(batch_size, 1, state["kv"].device)
            pos_q = self._expand_bom(
                (timestamps[:, tgt_frame_idx : tgt_frame_idx + 1],),
                self._num_samples,
                batch_size,
            )
            pos_k = self._expand_bom(
                (timestamps[:, : state["kv"].shape[1]],), self._num_samples, batch_size
            )
            z_hat = self.predictor(
                q,
                state["kv"],
                None,
                pos_q,
                pos_k,
                state.get("film_condition"),
                state.get("task_memory"),
                state.get("task_memory_mask"),
                step_action,
            )
            y_hat = self.tokenizer.decode(z_hat, state["x"], state["rope"])
            state["kv"] = torch.cat([state["kv"], z_hat.detach()], 1)
            state["x"] = y_hat.detach()
            return y_hat, state["y"][:, tgt_frame_idx], state

        batch_samples = batch_size * self._num_samples
        q_base = self._rollout_queries(batch_size, 1, state["kv"].device)
        q = q_base.expand(batch_samples, num_views, -1)
        q = self._add_camera_embedding(q)

        pos_q_time = timestamps[:, tgt_frame_idx : tgt_frame_idx + 1]
        pos_q = self._expand_bom(
            (pos_q_time.repeat_interleave(num_views, 1),),
            self._num_samples,
            batch_size,
        )
        num_key_frames = state["kv"].shape[1] // num_views
        pos_k_time = timestamps[:, :num_key_frames].repeat_interleave(num_views, 1)
        pos_k = self._expand_bom((pos_k_time,), self._num_samples, batch_size)

        z_hat = self.predictor(
            q,
            state["kv"],
            None,
            pos_q,
            pos_k,
            state.get("film_condition"),
            state.get("task_memory"),
            state.get("task_memory_mask"),
            None if step_action is None else step_action.repeat_interleave(num_views, 1),
        )
        x_flat = state["x"].reshape(batch_samples * num_views, *state["x"].shape[2:])
        y_hat_flat = self.tokenizer.decode(
            z_hat.reshape(batch_samples * num_views, 1, -1), x_flat, state["rope"]
        )
        y_hat = y_hat_flat.reshape(batch_samples, num_views, *y_hat_flat.shape[1:])

        kv_hat = self._add_camera_embedding(z_hat.reshape(batch_samples, num_views, -1))
        state["kv"] = torch.cat([state["kv"], kv_hat.detach()], 1)
        state["x"] = y_hat.detach()

        return y_hat, state["y"][:, tgt_frame_idx], state

    @torch.no_grad
    def _encode_frames(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if frames.ndim == 5:
            y = self.tokenizer.backbone(frames)
            x = self.tokenizer.backbone(torch.zeros_like(frames[:, :1]))[:, 0]
            rope = self.tokenizer._rope(frames)
            z = self.tokenizer.tokenize_offline(x, y, rope)
            return z.squeeze(2), y, rope
        if frames.ndim != 6:
            raise ValueError(f"frames must be [B,T,C,H,W] or [B,T,V,C,H,W], got {frames.shape}")

        batch, num_frames, num_views = frames.shape[:3]
        flat = frames.permute(0, 2, 1, 3, 4, 5).reshape(
            batch * num_views, num_frames, *frames.shape[3:]
        )
        y_flat = self.tokenizer.backbone(flat)
        # Black-frame reference is computed independently per physical camera
        # stream before camera embeddings are introduced.
        x_flat = self.tokenizer.backbone(torch.zeros_like(flat[:, :1]))[:, 0]
        rope = self.tokenizer._rope(flat)
        z_flat = self.tokenizer.tokenize_offline(x_flat, y_flat, rope).squeeze(2)
        z = z_flat.reshape(batch, num_views, num_frames, -1).permute(0, 2, 1, 3)
        y = y_flat.reshape(batch, num_views, num_frames, y_flat.shape[2], y_flat.shape[3]).permute(0, 2, 1, 3, 4)
        return z, y, rope
