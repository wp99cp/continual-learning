from avalanche.training.templates import SupervisedTemplate

from geometric_aware_sampling.experiments.base_experiment import BaseExperimentStrategy
from geometric_aware_sampling.experiments.geometric_aware_sampling.geometric_plugin import (
    GeometricPlugin,
)
from geometric_aware_sampling.experiments.geometric_aware_sampling.geometric_sampling_strategy import (
    RandomSamplingStrategy,
    DistributionWeightedSamplingStrategy,
    DistanceWeightedSamplingStrategy,
    DistanceWeightedScattering,
    DistanceWeightedSamplingStrategy_KL,
    NearestNeighborSamplingStrategy,
    DistanceWeightedSamplingStrategyExp,
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


class GeoAware_Baseline_1_WithoutGoldilock(BaseExperimentStrategy):
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
                    upper_quantile=1,
                    lower_quantile=0,
                    # we use all samples in the buffer,
                    # random sampling is done during mini-batch creation
                    # this should be equivalent to sampling the correct number of
                    # samples here, then we have no selection during mini-batch creation
                    p=P,  # TODO: tune per dataset to ensure unique samples
                ),
                # RepresentationPlugin(),
            ],
        )


class GeoAware_WeightedSampling_WithoutGoldilock(BaseExperimentStrategy):
    """

    Implements the Geometric Aware Sampling continual learning strategy
    as described in our paper with weighted sampling according to the distance of
    means.

    """

    def create_cl_strategy(self):

        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=DistanceWeightedSamplingStrategy,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    lower_quantile=0,
                    upper_quantile=1,
                    q=Q,
                    p=P,  # TODO: tune per dataset to ensure unique samples
                ),
                # RepresentationPlugin(),
            ],
        )


class GeoAware_WeightedSamplingExp_WithoutGoldilock(BaseExperimentStrategy):
    """

    Implements the Geometric Aware Sampling continual learning strategy
    as described in our paper with weighted sampling according to the distance of
    means.

    """

    def create_cl_strategy(self):

        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=DistanceWeightedSamplingStrategyExp,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    lower_quantile=0,
                    upper_quantile=1,
                    q=Q,
                    p=P,  # TODO: tune per dataset to ensure unique samples
                ),
                # RepresentationPlugin(),
            ],
        )


class GeoAware_KL_WithoutGoldilock(BaseExperimentStrategy):
    """

    Implements the Geometric Aware Sampling continual learning strategy
    as described in our paper with weighted sampling according to the distance of
    means.

    """

    def create_cl_strategy(self):

        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=DistanceWeightedSamplingStrategy_KL,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    lower_quantile=0,
                    upper_quantile=1,
                    q=Q,
                    p=P,  # TODO: tune per dataset to ensure unique samples
                ),
                # RepresentationPlugin(),
            ],
        )


class GeoAware_Scattering_WithoutGoldilock(BaseExperimentStrategy):
    """

    Implements the Geometric Aware Sampling continual learning strategy
    as described in our paper with weighted sampling according to the distance of
    means.

    """

    def create_cl_strategy(self):

        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=DistanceWeightedScattering,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    lower_quantile=0,
                    upper_quantile=1,
                    q=Q,
                    p=P,  # TODO: tune per dataset to ensure unique samples
                ),
                # RepresentationPlugin(),
            ],
        )


class GeoAware_NearestNeighbor_WithoutGoldilock(BaseExperimentStrategy):
    """

    Implements the Geometric Aware Sampling continual learning strategy
    as described in our paper with weighted sampling according to the distance of
    means.

    """

    def create_cl_strategy(self):

        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=NearestNeighborSamplingStrategy,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    lower_quantile=0,
                    upper_quantile=1,
                    q=Q,
                    p=P,  # TODO: tune per dataset to ensure unique samples
                ),
                # RepresentationPlugin(),
            ],
        )


class GeoAware_Baseline_1(BaseExperimentStrategy):
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
                    # we use all samples in the buffer,
                    # random sampling is done during mini-batch creation
                    # this should be equivalent to sampling the correct number of
                    # samples here, then we have no selection during mini-batch creation
                    p=P,  # TODO: tune per dataset to ensure unique samples
                ),
                # RepresentationPlugin(),
            ],
        )


class GeoAware_WeightedSampling(BaseExperimentStrategy):
    """

    Implements the Geometric Aware Sampling continual learning strategy
    as described in our paper with weighted sampling according to the distance of
    means.

    """

    def create_cl_strategy(self):

        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=DistanceWeightedSamplingStrategy,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    q=Q,
                    p=P,  # TODO: tune per dataset to ensure unique samples
                ),
                # RepresentationPlugin(),
            ],
        )


class GeoAware_DistributionWeightedSampling(BaseExperimentStrategy):
    def create_cl_strategy(self):

        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=DistributionWeightedSamplingStrategy,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    q=Q,
                    p=P,  # TODO: tune per dataset to ensure unique samples
                ),
                # RepresentationPlugin(),
            ],
        )


class GeoAware_WeightedSamplingExp(BaseExperimentStrategy):
    """

    Implements the Geometric Aware Sampling continual learning strategy
    as described in our paper with weighted sampling according to the distance of
    means.

    """

    def create_cl_strategy(self):

        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=DistanceWeightedSamplingStrategyExp,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    q=Q,
                    p=P,  # TODO: tune per dataset to ensure unique samples
                ),
                # RepresentationPlugin(),
            ],
        )


class GeoAware_KL(BaseExperimentStrategy):
    """

    Implements the Geometric Aware Sampling continual learning strategy
    as described in our paper with weighted sampling according to the distance of
    means.

    """

    def create_cl_strategy(self):

        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=DistanceWeightedSamplingStrategy_KL,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    q=Q,
                    p=P,  # TODO: tune per dataset to ensure unique samples
                ),
                # RepresentationPlugin(),
            ],
        )


class GeoAware_Scattering(BaseExperimentStrategy):
    """

    Implements the Geometric Aware Sampling continual learning strategy
    as described in our paper with weighted sampling according to the distance of
    means.

    """

    def create_cl_strategy(self):

        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=DistanceWeightedScattering,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    q=Q,
                    p=P,  # TODO: tune per dataset to ensure unique samples
                ),
                # RepresentationPlugin(),
            ],
        )


class GeoAware_NearestNeighbor(BaseExperimentStrategy):
    """

    Implements the Geometric Aware Sampling continual learning strategy
    as described in our paper with weighted sampling according to the distance of
    means.

    """

    def create_cl_strategy(self):

        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    sampling_strategy=NearestNeighborSamplingStrategy,
                    replay_ratio=REPLAY_RATIO,
                    mem_size=MAX_MEMORY_SIZE,
                    q=Q,
                    p=P,  # TODO: tune per dataset to ensure unique samples
                ),
                # RepresentationPlugin(),
            ],
        )
