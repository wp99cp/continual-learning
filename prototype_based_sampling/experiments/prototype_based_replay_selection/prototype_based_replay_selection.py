from avalanche.training.templates import SupervisedTemplate

from prototype_based_sampling.experiments.base_experiment import BaseExperimentStrategy
from prototype_based_sampling.experiments.prototype_based_replay_selection.prototype_plugin import (
    GeometricPlugin,
)
from prototype_based_sampling.experiments.prototype_based_replay_selection.prototype_based_replay_selection_strategy import (
    RandomSamplingWithEqualClassWeights,
    InvertedDistanceWeightedSampling,
    NegativeExponentialDistanceWeighted,
    WithingClassMaxScatter,
)

# full cifar100 dataset (however this is just a theoretical value)
# we store less samples per task in the buffer (see q value inside the plugin)
MAX_MEMORY_SIZE = 100_000

# ratio of training samples to keep in the buffer per task
# e.g. if task0 contains 100 samples split among 10 classes
# we keep 40 samples in the buffer split among all 10 classes
Q = 0.40

# ratio of new samples / buffer samples to use
# setting replay_ration = 1.0 will double the batch size
# for task_idx > 1, e.g. if batch_size is set to 64, we
# have for task_idx > 1 a batch size of 128 with 64 new samples
# and 64 buffer samples (replay samples)
# this was the default before the introduction of the replay_ratio
REPLAY_RATIO = 8.0 / 64  # that is 8 samples per mini-batch

# for cifar100 --> floor((100 / num_tasks) * 550 / 56) * 8
P = 1424


class Baseline_Random_Random(BaseExperimentStrategy):
    """
    Baseline with random buffer population and random replay sampling selection
    with equal amounts of samples per class.
    """

    def create_cl_strategy(self):
        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=RandomSamplingWithEqualClassWeights,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    lower_quantile=0,
                    upper_quantile=1,
                    q=Q,
                    p=P,
                ),
            ],
        )


class Baseline_Goldilocks_Random(BaseExperimentStrategy):
    """
    Baseline with buffer population based on the paper
    "Forgetting Order of Continual Learning" (Goldilocks strategy).

    Replay selection based on random sampling with equal amounts of samples per class.

    """

    def create_cl_strategy(self):
        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=RandomSamplingWithEqualClassWeights,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    q=Q,
                    p=P,
                ),
            ],
        )


class Baseline_Icarl_Random(BaseExperimentStrategy):
    """
    Baseline with buffer population based on iCarl strategy.
    Replay selection based on random sampling with equal amounts of samples per class.
    """

    def create_cl_strategy(self):
        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=RandomSamplingWithEqualClassWeights,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    q=Q,
                    p=1424,
                    storage_policy="ICarl",
                ),
            ],
        )


class PrototypeBased_Goldilocks_MaxScatter(BaseExperimentStrategy):
    """
    Prototype based strategy with buffer population based on the paper
    "Forgetting Order of Continual Learning" (Goldilocks strategy).

    Within class selection based on the maximum scatter between prototypes.
    """

    def create_cl_strategy(self):
        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=WithingClassMaxScatter,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    q=Q,
                    p=P,
                ),
            ],
        )


class PrototypeBased_Goldilocks_InvertedDistance(BaseExperimentStrategy):
    """

    Prototype based strategy with buffer population based on the paper
    "Forgetting Order of Continual Learning" (Goldilocks strategy).

    Random within class sampling with class weights distributed by the
    inverted distance between centroids of prototypes.

    """

    def create_cl_strategy(self):
        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=InvertedDistanceWeightedSampling,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    q=Q,
                    p=P,
                ),
            ],
        )


class PrototypeBased_Goldilocks_ExponentialDistance(BaseExperimentStrategy):
    """
    Prototype based strategy with buffer population based on the paper
    "Forgetting Order of Continual Learning" (Goldilocks strategy).

    Random within class sampling with class weights distributed by the
    negative exponential distance between centroids of prototypes.
    """

    def create_cl_strategy(self):
        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=NegativeExponentialDistanceWeighted,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    q=Q,
                    p=P,
                ),
            ],
        )
