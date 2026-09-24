"""Natural-language instruction conditioning for the action policy."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, T5EncoderModel


class T5InstructionEncoder(nn.Module):
    """Encode free-form instructions with T5 and project to action text tokens.

    The pretrained encoder is frozen by default. Its contextualized subword
    tokens are projected by a trainable linear layer into ActionDiT's text
    space. Keeping all visible T5 tokens lets cross-attention select instruction
    details instead of compressing the entire command into one task ID.
    """

    def __init__(
        self,
        model_name_or_path: str = "google-t5/t5-base",
        output_dim: int = 256,
        max_length: int = 64,
        freeze_encoder: bool = True,
        *,
        tokenizer: Any | None = None,
        text_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if output_dim <= 0 or max_length <= 0:
            raise ValueError("output_dim and max_length must be positive")
        self.model_name_or_path = str(model_name_or_path)
        self.max_length = int(max_length)
        self.freeze_encoder = bool(freeze_encoder)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.model_name_or_path
        )
        self.text_encoder = text_encoder or T5EncoderModel.from_pretrained(
            self.model_name_or_path
        )
        hidden_size = int(self.text_encoder.config.d_model)
        self.projection = nn.Linear(hidden_size, output_dim)
        if self.freeze_encoder:
            self.text_encoder.requires_grad_(False).eval()

    def train(self, mode: bool = True) -> "T5InstructionEncoder":
        super().train(mode)
        if self.freeze_encoder:
            # Frozen T5 must stay deterministic, especially its dropout layers.
            self.text_encoder.eval()
        return self

    @staticmethod
    def _normalize_instructions(instructions: str | Sequence[str]) -> list[str]:
        if isinstance(instructions, str):
            instructions = [instructions]
        else:
            instructions = list(instructions)
        if not instructions:
            raise ValueError("instruction batch cannot be empty")
        if any(not isinstance(text, str) or not text.strip() for text in instructions):
            raise ValueError("every instruction must be a non-empty string")
        return instructions

    def forward(
        self, instructions: str | Sequence[str]
    ) -> tuple[Tensor, Tensor]:
        instructions = self._normalize_instructions(instructions)
        encoded = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoder_device = next(self.text_encoder.parameters()).device
        input_ids = encoded["input_ids"].to(encoder_device)
        attention_mask = encoded["attention_mask"].to(encoder_device)
        grad_context = torch.no_grad() if self.freeze_encoder else nullcontext()
        with grad_context:
            hidden = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
        projection_param = self.projection.weight
        hidden = hidden.to(
            device=projection_param.device, dtype=projection_param.dtype
        )
        tokens = self.projection(hidden)
        mask = attention_mask.to(
            device=projection_param.device, dtype=torch.bool
        )
        return tokens, mask
