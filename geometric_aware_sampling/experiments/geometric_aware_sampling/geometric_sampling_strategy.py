import random
from abc import ABC

import numpy as np
import torch
from avalanche.benchmarks import AvalancheDataset
from avalanche.benchmarks.utils import concat_datasets
from avalanche.training import BalancedExemplarsBuffer

from torch.utils.data import DataLoader

from geometric_aware_sampling.models.SlimResNet18 import ResNet


class BufferSamplingStrategy(ABC):

    def __init__(self, balanced_buffer: BalancedExemplarsBuffer):
        self.balanced_buffer = balanced_buffer

    def sample(
        self,
        replay_size: int,
        _num_exps: int,
        current_model: torch.nn.Module | None = None,
        current_exp_dataset: AvalancheDataset = None,
    ) -> AvalancheDataset:
        """
        Sample elements from the buffer.
        :param current_model: the current model.
        :param replay_size: Number of samples to draw.
        :param _num_exps: Number of experiences.
        :param current_exp_dataset: the current experience dataset.
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
        current_exp_dataset: AvalancheDataset = None,
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
        current_exp_dataset: AvalancheDataset = None,
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


class DistanceWeightedSamplingStrategy(BufferSamplingStrategy):
    """
    Strategy, that samples based on the inverted square distance of the prototypes
    """

    def __init__(self, balanced_buffer: BalancedExemplarsBuffer):
        super().__init__(balanced_buffer)
        self.num_exp = 0
        self.ratios = dict()

    def sample(
        self,
        replay_size: int,
        _num_exps: int,
        current_model: torch.nn.Module | None = None,
        current_exp_dataset: AvalancheDataset = None,
    ) -> AvalancheDataset:

        if not isinstance(current_model, ResNet):
            raise Exception("Cannot sample without access to the last layer.")

        if _num_exps > self.num_exp:
            self.before_experience(current_model, current_exp_dataset)

        concatenated = concat_datasets(
            [
                b.buffer.subset(
                    np.random.choice(
                        len(b.buffer),
                        size=int(replay_size * self.ratios[c]),
                        replace=True,
                    )
                )
                for c, b in self.balanced_buffer.buffer_groups.items()
            ]
        )
        print("Size of concat dataset is " + str(len(concatenated)))
        return concatenated

    def before_experience(
        self, current_model: ResNet, current_exp_dataset: AvalancheDataset
    ):
        """
        Computes representation means of all classes. From them, it calculates
        the ratio based on the inverse of the distance.
        """
        means = dict()
        inv_distances = dict()

        for c, b in self.balanced_buffer.buffer_groups.items():
            # computing mean for class c
            loader = DataLoader(b.buffer, batch_size=len(b.buffer), shuffle=False)
            # Complicated way of getting the data out
            for all_data in loader:
                pass
            representations = current_model.extract_last_layer(all_data[0])
            means[c] = representations.mean(dim=0)
            inv_distances[c] = 0

        print("Means for learned classes calculated")

        for c, ids in current_exp_dataset.targets.val_to_idx.items():
            curr_dataset = current_exp_dataset.subset(ids)
            loader = DataLoader(
                curr_dataset, batch_size=len(curr_dataset), shuffle=False
            )
            # Complicated way of getting the data out
            for all_data in loader:
                pass
            current_exp_mean = current_model.extract_last_layer(all_data[0]).mean(dim=0)
            # compute norm
            w_c = dict()
            for c_old, mean_old in means.items():
                w_c[c_old] = 1 / torch.norm((current_exp_mean - mean_old)).item()
            s = sum(w_c.values())
            for c_old, mean_old in means.items():
                inv_distances[c_old] += w_c[c_old] / s

        for c, inv_dist in inv_distances.items():
            # sum(inv_distances.values()) == n_classes
            self.ratios[c] = inv_dist / sum(inv_distances.values())

        print("Ratios are" + str(self.ratios.values()))
