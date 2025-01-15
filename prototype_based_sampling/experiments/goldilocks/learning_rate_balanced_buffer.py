import torch
from avalanche.training import BalancedExemplarsBuffer
from avalanche.training.templates import SupervisedTemplate

from prototype_based_sampling.experiments.goldilocks.learning_speed_plugin import (
    LearningSpeedPlugin,
)
from prototype_based_sampling.experiments.goldilocks.weighted_sampling_buffer import (
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


class LearningRateBalancedBuffer(BalancedExemplarsBuffer[WeightedSamplingBuffer]):
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
        p: float = 0.25,
        s: float = 0.75,
    ):
        """
        :param max_size: max number of total input samples in the replay
            memory.
        :param adaptive_size: True if mem_size is divided equally over all
                              observed experiences (keys in replay_mem).
        :param num_experiences: If adaptive size is False, the fixed number
                                of experiences to divide capacity over.

        :param p: the upper quantile of the learning speed distribution that will
                  never be included in the buffer

        :param s: the lower quantile of the learning speed distribution that will
                never be included in the buffer
        """
        super().__init__(max_size, adaptive_size, num_experiences)
        self._num_exps = 0
        self.p = p
        self.q = s

    def post_adapt(self, strategy: "SupervisedTemplate", exp):
        self._num_exps += 1
        new_data = exp.dataset
        lens = self.get_group_lengths(self._num_exps)

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

        # create a mask, which masks the top q% and bottom s% of the learning speed
        mask = (learning_speed > learning_speed.quantile(self.q)) & (
            learning_speed < learning_speed.quantile(self.p)
        )
        weights[mask] = (
            0  # set the weights of the samples in the mask to 0, so they are not included in the buffer
        )

        new_buffer.update_from_dataset(new_data, weights=weights)
        self.buffer_groups[self._num_exps - 1] = new_buffer

        for ll, b in zip(lens, self.buffer_groups.values()):
            b.resize(strategy, ll)
