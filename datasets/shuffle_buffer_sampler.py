"""Bounded shuffle-buffer sampler with DistributedDataParallel sharding."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator

import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler


def bounded_shuffle(
    indices: Iterator[int], *, buffer_size: int, rng: random.Random
) -> Iterator[int]:
    """Shuffle a stream using the bounded-buffer algorithm used by tf.data."""
    if buffer_size <= 0:
        raise ValueError("shuffle buffer size must be positive")

    source = iter(indices)
    buffer: list[int] = []
    for _ in range(buffer_size):
        try:
            buffer.append(next(source))
        except StopIteration:
            break

    while buffer:
        slot = rng.randrange(len(buffer))
        yield buffer[slot]
        try:
            buffer[slot] = next(source)
        except StopIteration:
            buffer.pop(slot)


class ShuffleBufferDistributedSampler(DistributedSampler):
    """Finite-epoch equivalent of OpenVLA-OFT frame-level RLDS shuffle.

    Every rank deterministically constructs the same global buffer-shuffled
    stream, then consumes its disjoint strided shard. In the uncommon case
    that the dataset size is not divisible by the world size, this follows
    PyTorch DistributedSampler and pads with at most world_size - 1 leading
    indices so every rank executes the same number of batches.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        buffer_size: int,
        seed: int = 0,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        if num_replicas is None:
            num_replicas = (
                dist.get_world_size()
                if dist.is_initialized()
                else int(os.environ.get("WORLD_SIZE", "1"))
            )
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else int(
                os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))
            )
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=int(seed),
            drop_last=False,
        )
        self.buffer_size = int(buffer_size)
        if self.buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be positive")

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        shuffled = bounded_shuffle(
            iter(range(len(self.dataset))),
            buffer_size=self.buffer_size,
            rng=rng,
        )

        padding_size = self.total_size - len(self.dataset)
        prefix: list[int] = []
        for global_position, index in enumerate(shuffled):
            if len(prefix) < padding_size:
                prefix.append(index)
            if global_position % self.num_replicas == self.rank:
                yield index

        for offset, index in enumerate(prefix):
            global_position = len(self.dataset) + offset
            if global_position % self.num_replicas == self.rank:
                yield index
