from typing import Type

import torch
from avalanche.benchmarks import AvalancheDataset
from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.training import BalancedExemplarsBuffer
from avalanche.training.templates import SupervisedTemplate
from torch.utils.data import DataLoader

from geometric_aware_sampling.experiments.geometric_aware_sampling.geometric_sampling_strategy import (
    RandomSamplingWithEqualClassWeights,
    BufferSamplingStrategy,
)
from geometric_aware_sampling.experiments.goldilocks.learning_speed_plugin import (
    LearningSpeedPlugin,
)
from geometric_aware_sampling.experiments.goldilocks.weighted_sampling_buffer import (
    WeightedSamplingBuffer,
)


def get_learning_speed(strategy: "SupervisedTemplate"):
    """
    Get the learning speed of the samples
    """

    learning_speed_plugin = None
    for plugin in strategy.plugins:
        if isinstance(plugin, LearningSpeedPlugin):
            learning_speed_plugin = plugin
            break

    return learning_speed_plugin.learning_speed


class GeometricBalancedBuffer(BalancedExemplarsBuffer[WeightedSamplingBuffer]):
    """Rehearsal buffer with samples balanced over experiences.

    The number of experiences can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed experiences so far.
    """

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = True,
        num_experiences=None,
        upper_quantile_ls: float = 0.25,
        lower_quantile_ls: float = 0.75,
        q: float = 0.4,
        p: float = 1.0,
        sampling_strategy: Type[
            BufferSamplingStrategy
        ] = RandomSamplingWithEqualClassWeights,
    ):
        """
        :param max_size: max number of total input samples in the replay
            memory.
        :param replay_batch_size: number of training samples from replay buffer pool.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.

        :param upper_quantile_ls: the upper quantile of the learning speed distribution that will
                  never be included in the buffer

        :param lower_quantile_ls: the lower quantile of the learning speed distribution that will
                never be included in the buffer

        :param q: ratio of training samples to keep
        :param p: ratio of buffer samples to use, if p > 1.0 we treat p as a fixed value for the number of samples
        :param sampling_strategy: the sampling strategy to use for replay (e.g. random or geometry based)
        """
        super().__init__(max_size, adaptive_size, num_experiences)
        self.pool_size = 0
        self._num_classes = 0
        self.upper_quantile_ls = upper_quantile_ls
        self.lower_quantile_ls = lower_quantile_ls
        self.q = q
        self.p = p

        self.replay_sampler = sampling_strategy(balanced_buffer=self)

    def get_group_lengths(self, total_size, num_groups):
        """Compute groups lengths given the number of groups `num_groups`."""
        if self.adaptive_size:
            lengths = [total_size // num_groups for _ in range(num_groups)]
            # distribute remaining size among experiences.
            rem = total_size - sum(lengths)
            for i in range(rem):
                lengths[i] += 1
        else:
            lengths = [total_size // self.total_num_groups for _ in range(num_groups)]
        return lengths

    @property
    def buffer(self):
        return self.get_buffer(None)

    def get_buffer(
        self,
        current_model: torch.nn.Module | None,
        experience_dataset: torch.utils.data.Dataset,
    ):

        # nothing to replay from
        if self._num_classes == 0:
            return concat_datasets([])

        # Sample p% of all elements in the buffer pool
        # or fixed size if p > 1.0
        replay_size = int(self.p)
        if self.p <= 1.0:
            replay_size = int(self.p * self.pool_size)

        print(
            f"Sampling {replay_size} samples from buffer using {self.replay_sampler.__class__.__name__}"
        )

        return self.replay_sampler.sample(
            replay_size=replay_size,
            _num_exps=self._num_classes,
            current_model=current_model,
            current_exp_dataset=experience_dataset,
        )

    def log_buffer_summary(
        self, kwargs, task_idx: int, buffer_name: str = "GeometricBalanced"
    ):
        for c, buffer in self.buffer_groups.items():
            buffer.log_buffer_summary(
                kwargs, task_idx, f"buffer_class_{c}--" + buffer_name
            )

    def post_adapt(self, strategy: "SupervisedTemplate", exp):
        new_data = exp.dataset
        lbls_tensor = torch.tensor([x[1] for x in new_data])
        unique_labels = torch.unique(lbls_tensor)
        self._num_classes += len(unique_labels)

        # Increase pool size by q% of all new samples
        print(f"Adding {int(self.q * len(new_data))} samples to the buffer")
        print(
            f" » that is {int(self.q * 100)}% of the current epoch dataset of length {len(new_data)}"
        )
        self.pool_size += int(self.q * len(new_data))
        self.pool_size = min(self.pool_size, self.max_size)

        lens = self.get_group_lengths(self.pool_size, self._num_classes)

        learning_speed = get_learning_speed(strategy)
        if learning_speed is None:
            raise ValueError("LearningSpeedPlugin not found in the strategy")

        # we initialize the weights of the samples to a random value
        # to ensure uniform sampling, we later set the weights of the samples
        # in the mask to 0 to exclude them from the buffer
        #
        # based on https://en.wikipedia.org/wiki/Reservoir_sampling
        weights = torch.rand(len(new_data))

        # create a mask, which masks the top and bottom of the learning speed
        print(
            f"Masking learning speed between {self.lower_quantile_ls} and {self.upper_quantile_ls}"
        )

        lower_bound = learning_speed.quantile(self.lower_quantile_ls)
        upper_bound = learning_speed.quantile(self.upper_quantile_ls)

        print(
            f" » thus we take learning_speed in [{lower_bound}, {upper_bound}] (including the boundaries)"
        )
        # the equal is necessary for sampling everything
        mask = (learning_speed <= lower_bound) | (learning_speed >= upper_bound)
        print(f" » {mask.sum()} samples masked")

        # create a histogram plot of the learning speed using matplotlib
        # and mark the bounds as vertical lines
        # fig, ax = plt.subplots()
        # ax.hist(learning_speed, bins=20)
        # ax.axvline(lower_bound, color="r", linestyle="--")
        # ax.axvline(upper_bound, color="r", linestyle="--")

        # save figure to tensorboard
        # fig.show()

        # weights of samples in mask to 0 -> not included in the buffer
        weights[mask] = 0

        for i, l in enumerate(unique_labels):
            # Initialize buffer for new samples with a certain size
            idx = self._num_classes - len(unique_labels) + i
            new_buffer = WeightedSamplingBuffer(lens[idx])
            # set elements of new buffer based on new_data and random masked weights
            mask_label = lbls_tensor != l
            weights_l = weights.clone()
            weights_l[mask_label] = 0
            new_buffer.update_from_dataset(new_data, weights=weights_l)
            # Index buffers by classes
            self.buffer_groups[l] = new_buffer

        # resize other buffers
        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(strategy, ll)


class GeometricBalancedBufferICarl(BalancedExemplarsBuffer[WeightedSamplingBuffer]):
    """Rehearsal buffer with samples balanced over experiences.
    The number of experiences can be fixed up front or adaptive, based on
    the 'adaptive_size' attribute. When adaptive, the memory is equally
    divided over all the unique observed experiences so far.
    """

    def __init__(
        self,
        max_size: int,
        adaptive_size: bool = True,
        num_experiences=None,
        upper_quantile_ls: float = 0.25,
        lower_quantile_ls: float = 0.75,
        q: float = 0.4,
        p: float = 1.0,
        sampling_strategy: Type[
            BufferSamplingStrategy
        ] = RandomSamplingWithEqualClassWeights,
    ):
        """
        :param max_size: max number of total input samples in the replay
            memory.
        :param replay_batch_size: number of training samples from replay buffer pool.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.
        :param upper_quantile_ls: the upper quantile of the learning speed distribution that will
                  never be included in the buffer
        :param lower_quantile_ls: the lower quantile of the learning speed distribution that will
                never be included in the buffer
        :param q: ratio of training samples to keep
        :param p: ratio of buffer samples to use, if p > 1.0 we treat p as a fixed value for the number of samples
        :param sampling_strategy: the sampling strategy to use for replay (e.g. random or geometry based)
        """
        super().__init__(max_size, adaptive_size, num_experiences)
        self.pool_size = 0
        self._num_classes = 0
        self.upper_quantile_ls = upper_quantile_ls
        self.lower_quantile_ls = lower_quantile_ls
        self.q = q
        self.p = p

        self.replay_sampler = sampling_strategy(balanced_buffer=self)

    def get_group_lengths(self, total_size, num_groups):
        """Compute groups lengths given the number of groups `num_groups`."""
        if self.adaptive_size:
            lengths = [total_size // num_groups for _ in range(num_groups)]
            # distribute remaining size among experiences.
            rem = total_size - sum(lengths)
            for i in range(rem):
                lengths[i] += 1
        else:
            lengths = [total_size // self.total_num_groups for _ in range(num_groups)]
        return lengths

    @property
    def buffer(self):
        return self.get_buffer(None)

    def get_buffer(
        self,
        current_model: torch.nn.Module | None,
        experience_dataset: AvalancheDataset,
    ):

        # nothing to replay from
        if self._num_classes == 0:
            return concat_datasets([])

        # Sample p% of all elements in the buffer pool
        # or fixed size if p > 1.0
        replay_size = int(self.p)
        if self.p <= 1.0:
            replay_size = int(self.p * self.pool_size)

        print(
            f"Sampling {replay_size} samples from buffer using {self.replay_sampler.__class__.__name__}"
        )

        return self.replay_sampler.sample(
            replay_size=replay_size,
            _num_exps=self._num_classes,
            current_model=current_model,
            current_exp_dataset=experience_dataset,
        )

    def log_buffer_summary(
        self, kwargs, task_idx: int, buffer_name: str = "GeometricBalanced"
    ):
        for c, buffer in self.buffer_groups.items():
            buffer.log_buffer_summary(
                kwargs, task_idx, f"buffer_class_{c}--" + buffer_name
            )

    def get_representation(self, model: torch.nn.Module, dataset):
        batch_size = 2048
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, pin_memory=False
        )
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

    def post_adapt(self, strategy: "SupervisedTemplate", exp):

        # Last layer
        new_data = exp.dataset

        lbls_tensor = torch.tensor([x[1] for x in new_data])
        unique_labels = torch.unique(lbls_tensor)
        self._num_classes += len(unique_labels)

        # Increase pool size by q% of all new samples
        print(f"Adding {int(self.q * len(new_data))} samples to the buffer")
        print(
            f" » that is {int(self.q * 100)}% of the current epoch dataset of length {len(new_data)}"
        )
        self.pool_size += int(self.q * len(new_data))
        self.pool_size = min(self.pool_size, self.max_size)

        lens = self.get_group_lengths(self.pool_size, self._num_classes)

        weights = torch.rand(len(new_data))

        # representations, raw_data = self.get_representation(strategy.model, new_data)

        # create new buffers
        for j, l in enumerate(unique_labels):
            ids = torch.nonzero(lbls_tensor == l)
            curr_dataset = new_data.subset(ids)
            representations = self.get_representation(strategy.model, curr_dataset)

            # Initialize buffer for new samples with a certain size
            idx = self._num_classes - len(unique_labels) + j

            # Number of new exemplars
            m = lens[idx]

            new_buffer = WeightedSamplingBuffer(m)

            new_data_size_l = len(ids)

            mean = representations.mean(dim=0)

            phi_sum = torch.zeros(representations.shape[1:])

            i, added, selected = 0, 0, []
            weights_l = weights.clone()
            weights_l[lbls_tensor != l] = 0
            weights_short = torch.zeros(new_data_size_l)

            while (not (added == m)) and (i < 20 * m):
                min_id = torch.argmin(
                    torch.norm(
                        mean.unsqueeze(0).expand(new_data_size_l, *mean.shape)
                        - (
                            phi_sum.unsqueeze(0).expand(new_data_size_l, *phi_sum.shape)
                            + representations
                        )
                        / (i + 1),
                        dim=-1,
                    ),
                    dim=0,
                ).item()
                if min_id not in selected:
                    weights_short[min_id] += m - added
                    added += 1
                    selected.append(min_id)
                # print("Minimal id is " + str(min_id))
                phi_sum += representations[min_id]
                i += 1

            weights_l[lbls_tensor == l] = weights_short
            new_buffer.update_from_dataset(new_data, weights=weights_l)
            # Index buffers by classes
            self.buffer_groups[l] = new_buffer

        # resize other buffers
        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(strategy, ll)
