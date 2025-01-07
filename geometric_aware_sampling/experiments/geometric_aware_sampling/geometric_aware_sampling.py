from avalanche.training.templates import SupervisedTemplate

from geometric_aware_sampling.experiments.base_experiment import BaseExperimentStrategy
from geometric_aware_sampling.experiments.geometric_aware_sampling.geometric_plugin import (
    GeometricPlugin,
)
from geometric_aware_sampling.experiments.geometric_aware_sampling.geometric_sampling_strategy import (
    RandomSamplingStrategy,
    RandomWeightedSamplingStrategy,
)
from geometric_aware_sampling.experiments.geometric_aware_sampling.representation_plugin import (
    RepresentationPlugin,
)

# full cifar100 dataset (however this is just a theoretical value)
# we store less samples per task in the buffer (see q value inside the plugin)
MAX_MEMORY_SIZE = 100_000

# ratio of training samples to keep in the buffer per task
# e.g. if task0 contains 100 samples split among 10 classes
# we keep 40 samples in the buffer split among all 10 classes
Q = 0.25

# ratio of new samples / buffer samples to use
# setting replay_ration = 1.0 will double the batch size
# for task_idx > 1, e.g. if batch_size is set to 64, we
# have for task_idx > 1 a batch size of 128 with 64 new samples
# and 64 buffer samples (replay samples)
# this was the default before the introduction of the replay_ratio
REPLAY_RATIO = 0.25  # that is 16 samples per mini-batch


class GeometricAwareSamplingStrategy__Baseline_1(BaseExperimentStrategy):
    """

    This strategy implements the baseline 1 for the Geometric Aware Sampling.
    We use learning rate based sampling for buffer population.

    We use random replay selection for the replay buffer.

    """

    def create_cl_strategy(self):

        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=RandomSamplingStrategy,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    q=Q,
                ),
                RepresentationPlugin(),
            ],
        )


class GeometricAwareSamplingStrategy(BaseExperimentStrategy):
    """

    Implements the Geometric Aware Sampling continual learning strategy
    as described in our paper

    """

    def create_cl_strategy(self):

        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=RandomWeightedSamplingStrategy,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    q=Q,
                ),
                RepresentationPlugin(),
            ],
        )
