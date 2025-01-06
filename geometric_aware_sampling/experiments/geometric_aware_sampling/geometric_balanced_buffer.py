import torch
import random
from avalanche.training import BalancedExemplarsBuffer
from avalanche.training.templates import SupervisedTemplate
from avalanche.benchmarks.utils.utils import concat_datasets

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
        :param p: ratio of buffer samples to use
        """
        super().__init__(max_size, adaptive_size, num_experiences)
        self.pool_size = 0
        self._num_exps = 0
        self.upper_quantile_ls = upper_quantile_ls
        self.lower_quantile_ls = lower_quantile_ls
        self.q = q
        self.p = p

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
        datasets = []
        if self._num_exps > 0:
            # Sample p% of all elements in the buffer pool
            replay_size = int(self.p * self.pool_size)
            # Sample evenly from all groups
            lens = self.get_group_lengths(replay_size, self._num_exps)

            for ll, b in zip(lens, self.buffer_groups.values()):
                buf = b.buffer
                l = min(ll, len(buf))
                # Sample l random indices without replacement
                indices = random.sample(range(len(buf)), k=l)
                datasets.append(buf.subset(indices))
        return concat_datasets(datasets)

    def post_adapt(self, strategy: "SupervisedTemplate", exp):
        self._num_exps += 1
        new_data = exp.dataset

        # Increase pool size by q% of all new samples
        self.pool_size += int(self.q * len(new_data))
        self.pool_size = min(self.pool_size, self.max_size)

        lens = self.get_group_lengths(self.pool_size, self._num_exps)

        # Initialize buffer for new samples with a certain size
        new_buffer = WeightedSamplingBuffer(lens[-1])

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
        mask = (learning_speed > learning_speed.quantile(self.lower_quantile_ls)) & (
            learning_speed < learning_speed.quantile(self.upper_quantile_ls)
        )
        weights[mask] = (
            0  # weights of samples in mask to 0 -> not included in the buffer
        )

        # set elements of new buffer based on new_data and random masked weights
        new_buffer.update_from_dataset(new_data, weights=weights)
        self.buffer_groups[self._num_exps - 1] = new_buffer

        # resize other buffers
        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(strategy, ll)
