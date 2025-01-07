import random
from abc import ABC

import numpy as np
import torch
from avalanche.benchmarks import AvalancheDataset
from avalanche.benchmarks.utils import concat_datasets
from avalanche.training import BalancedExemplarsBuffer


class BufferSamplingStrategy(ABC):

    def __init__(self, balanced_buffer: BalancedExemplarsBuffer):
        self.balanced_buffer = balanced_buffer

    def sample(
        self,
        replay_size: int,
        _num_exps: int,
        current_model: torch.nn.Module | None = None,
    ) -> AvalancheDataset:
        """
        Sample elements from the buffer.
        :param current_model: the current model.
        :param replay_size: Number of samples to draw.
        :param _num_exps: Number of experiences.
        :return: A dataset with the sampled elements.
        """
        raise Exception("Not implemented")

    @property
    def complete_buffer(self):
        """
        Concatenates all buffer groups into a single buffer.
        This is useful for sampling strategies that do not require
        balancing the number of samples per experience.
        """

        buffer = concat_datasets(
            [b.buffer for b in self.balanced_buffer.buffer_groups.values()]
        )
        return buffer


class RandomSamplingStrategy(BufferSamplingStrategy):
    """
    Strategy that samples randomly from the buffer while balancing the
    number of samples per experience. This is used for baseline 1.
    """

    def sample(
        self,
        replay_size: int,
        _num_exps: int,
        current_model: torch.nn.Module | None = None,
    ) -> AvalancheDataset:
        datasets = []

        # Sample evenly from all groups
        lens = self.balanced_buffer.get_group_lengths(replay_size, _num_exps)

        for ll, b in zip(lens, self.balanced_buffer.buffer_groups.values()):
            buf = b.buffer
            l = min(ll, len(buf))
            # Sample l random indices without replacement
            indices = random.sample(range(len(buf)), k=l)
            datasets.append(buf.subset(indices))

        return concat_datasets(datasets)


class RandomWeightedSamplingStrategy(BufferSamplingStrategy):
    """
    Strategy that samples randomly from the buffer while balancing the
    number of samples per experience. This is used for the geometric aware
    sampling strategy.
    """

    def sample(
        self,
        replay_size: int,
        _num_exps: int,
        current_model: torch.nn.Module | None = None,
    ) -> AvalancheDataset:

        buffer = self.complete_buffer
        print(f"sampling {replay_size} from a buffer with {len(buffer)} samples.")

        # Ensure the buffer has enough elements
        if len(buffer) < replay_size:
            raise ValueError(
                f"Not enough elements in the buffer to sample {replay_size} elements."
            )

        weights = (np.arange(len(buffer))[::-1] + 1) ** 2
        probabilities = weights / weights.sum()

        # Sample indices based on the exponential probabilities
        indices = np.random.choice(
            len(buffer), size=replay_size, replace=False, p=probabilities
        )

        return buffer.subset(indices)
