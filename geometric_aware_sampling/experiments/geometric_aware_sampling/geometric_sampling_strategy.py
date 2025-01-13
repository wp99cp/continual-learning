import math
from abc import ABC

import numpy as np
import torch
from avalanche.benchmarks import AvalancheDataset
from avalanche.benchmarks.utils import concat_datasets
from avalanche.training import BalancedExemplarsBuffer
import torch.distributions.multivariate_normal
from torch.masked import softmax, log_softmax
from torch.utils.data import DataLoader

from geometric_aware_sampling.models.SlimResNet18 import ResNet


class BufferSamplingStrategy(ABC):

    def __init__(self, balanced_buffer: BalancedExemplarsBuffer):
        self.balanced_buffer = balanced_buffer
        self.num_exp = 0
        self.ratios = dict()

    def before_experience(
        self, current_model: ResNet, current_exp_dataset: AvalancheDataset
    ):
        """
        This method is called before a new experience is encountered.
        It can be used to update the ratios based on the current model and dataset.
        """
        raise NotImplementedError

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
        buffer = self.complete_buffer
        print(f"sampling {replay_size} from a buffer with {len(buffer)} samples.")

        if not isinstance(current_model, ResNet):
            raise Exception("Cannot sample without access to the last layer.")

        time_before = torch.cuda.Event(enable_timing=True)
        time_before.record()

        if _num_exps > self.num_exp:
            self.before_experience(current_model, current_exp_dataset)

        print("\nRatios are " + str(self.ratios.values()), end="\n\n")
        print(f" » Ratios sum up to {sum(self.ratios.values())}")
        print(
            f" » Ratios {np.mean(list(self.ratios.values()))} +- {np.std(list(self.ratios.values()))}"
        )
        print(
            f" » min/max {min(self.ratios.values())} / {max(self.ratios.values())}",
            end="\n\n",
        )

        print(
            f" » This we sample {[
            int(replay_size * r) for r in self.ratios.values()
        ]} samples per class for classes {list(self.ratios.keys())} with replace {
        [len(buffer) < int(replay_size * r) for r in self.ratios.values()]
        }"
        )

        concatenated = concat_datasets(
            [
                b.buffer.subset(
                    np.random.choice(
                        len(b.buffer),
                        size=int(replay_size * self.ratios[c]),
                        replace=len(b.buffer) < int(replay_size * self.ratios[c]),
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

        buffer = self.complete_buffer
        idxs = np.random.choice(
            len(buffer),
            size=int(replay_size),
            replace=len(buffer) < int(replay_size),
        )
        return buffer.subset(idxs)


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

class DistributionWeightedSamplingStrategy(BufferSamplingStrategy):
    """
    Strategy, that samples based on the gaussian mixture distribution of prototypes
    """

    def before_experience(
        self, current_model: ResNet, current_exp_dataset: AvalancheDataset
    ):
        """
        Computes representation means of all classes. From them, it calculates
        the ratio based on the inverse of the distance.
        """
        dists = dict()
        mean_densities = dict()

        for c, b in self.balanced_buffer.buffer_groups.items():
            representations = get_representation(current_model, b.buffer)
            mean = representations.mean(dim=0)
            sigma = torch.cov(representations.T)
            dists[c] = torch.distributions.MultivariateNormal(mean, sigma)
            mean_densities[c] = 0

        for c, ids in current_exp_dataset.targets.val_to_idx.items():
            curr_dataset = current_exp_dataset.subset(ids)
            representations = get_representation(current_model, curr_dataset)

            # compute norm
            w_c = dict()
            for c_old, dist in dists.items():
                w_c[c_old] = torch.exp(dist.log_prob()).item()

            s = sum(w_c.values())
            for c_old in dists.keys():
                mean_densities[c_old] += w_c[c_old] / s

        for c, mean_density in mean_densities.items():
            # sum(inv_distances.values()) == n_classes
            self.ratios[c] = mean_density / sum(mean_densities.values())

class DistanceWeightedSamplingStrategy(BufferSamplingStrategy):
    """
    Strategy, that samples based on the inverted square distance of the prototypes
    """

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


class DistanceWeightedSamplingStrategy_KL(BufferSamplingStrategy):

    def before_experience(
        self, current_model: ResNet, current_exp_dataset: AvalancheDataset
    ):
        """
        Computes representation means of all classes. From them, it calculates
        the ratio based on the inverse of the distance.
        """
        distances = dict()
        representations_old = dict()
        self.ratios = dict()

        for c, b in self.balanced_buffer.buffer_groups.items():
            # fill with zeros
            distances[c] = 0
            representations_old[c] = get_representation(current_model, b.buffer)

        for c, ids in current_exp_dataset.targets.val_to_idx.items():
            curr_dataset = current_exp_dataset.subset(ids)
            representations = get_representation(current_model, curr_dataset)

            # compute norm
            kl_div_loss = torch.nn.KLDivLoss(reduction="mean")
            for c_old, old_rep in representations_old.items():
                distances[c_old] += math.exp(
                    -kl_div_loss(
                        log_softmax(old_rep, dim=0),
                        softmax(representations, dim=0),
                    ).item()
                )

        # boost the rations with exp(-distances)
        for c, dist in distances.items():
            self.ratios[c] = dist

        # normalize
        normalization_factor = sum(self.ratios.values())
        for c, dist in distances.items():
            self.ratios[c] = self.ratios[c] / normalization_factor


class DistanceWeightedSamplingStrategyExp(BufferSamplingStrategy):
    """
    Strategy, that samples based on the inverted square distance of the prototypes
    """

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
                w_c[c_old] = torch.exp(-torch.norm(current_exp_mean - mean_old)).item()

            s = sum(w_c.values())
            for c_old, mean_old in means.items():
                inv_distances[c_old] += w_c[c_old] / s

        for c, inv_dist in inv_distances.items():
            # sum(inv_distances.values()) == n_classes
            self.ratios[c] = inv_dist / sum(inv_distances.values())


class DistanceWeightedScattering(BufferSamplingStrategy):
    """
    Strategy, that samples based on the inverted square distance of the prototypes
    """

    def before_experience(
        self, current_model: ResNet, current_exp_dataset: AvalancheDataset
    ):
        """
        Computes representation means of all classes. From them, it calculates
        the ratio based on the inverse of the distance.
        """

        for c, b in self.balanced_buffer.buffer_groups.items():
            representations = get_representation(current_model, b.buffer)
            self.ratios[c] = representations.std().item()

        # normalize
        s = sum(self.ratios.values())
        for c in self.ratios.keys():
            self.ratios[c] = self.ratios[c] / s


class NearestNeighborSamplingStrategy(BufferSamplingStrategy):

    def sample(
        self,
        replay_size: int,
        _num_exps: int,
        current_model: torch.nn.Module | None = None,
        current_exp_dataset: AvalancheDataset = None,
    ) -> AvalancheDataset:

        buffer = self.complete_buffer

        if not isinstance(current_model, ResNet):
            raise Exception("Cannot sample without access to the last layer.")

        representations_buffer = get_representation(current_model, buffer)
        representation_new = get_representation(current_model, current_exp_dataset)

        # compute distances
        distances = torch.cdist(representation_new, representations_buffer)

        idxs = distances.argsort(dim=1)[:, :replay_size]
        return buffer.subset(idxs.flatten())
