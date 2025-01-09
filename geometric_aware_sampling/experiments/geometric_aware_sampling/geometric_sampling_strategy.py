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


def get_representation(model: torch.nn.Module, dataset):
    batch_size = 2048
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
    representations = []  # representation of the dataset (stored on CPU)

    # set model to eval mode
    model_mode = model.training
    model.eval()

    # run inference on the GPU
    for batch_data in loader:
        # dim 0 is the data dimension, dim 1 is the target dimension, ...
        gpu_batch = batch_data[0].to("cuda")
        batch_representations = model.extract_last_layer(gpu_batch)
        representations.append(batch_representations.detach().cpu())

        # Explicitly delete GPU tensors
        del gpu_batch, batch_representations
        torch.cuda.empty_cache()

    # set model back to its original mode
    model.train(model_mode)
    return torch.cat(representations, dim=0)


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

        buffer = self.complete_buffer
        print(f"sampling {replay_size} from a buffer with {len(buffer)} samples.")

        if not isinstance(current_model, ResNet):
            raise Exception("Cannot sample without access to the last layer.")

        time_before = torch.cuda.Event(enable_timing=True)
        time_before.record()

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

        time_after = torch.cuda.Event(enable_timing=True)
        time_after.record()
        torch.cuda.synchronize()
        print(f"Time to sample {time_before.elapsed_time(time_after):.2f} ms")
        print("Size of concat dataset is " + str(len(concatenated)))

        # empty cuda cache
        torch.cuda.empty_cache()
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
            representations = get_representation(current_model, b.buffer)
            means[c] = representations.mean(dim=0)
            inv_distances[c] = 0

        for c, ids in current_exp_dataset.targets.val_to_idx.items():
            curr_dataset = current_exp_dataset.subset(ids)
            representations = get_representation(current_model, curr_dataset)
            current_exp_mean = representations.mean(dim=0)

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

        print("Ratios are " + str(self.ratios.values()), end="\n\n")
