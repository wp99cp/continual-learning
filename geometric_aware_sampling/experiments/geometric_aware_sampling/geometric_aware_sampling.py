from avalanche.training.templates import SupervisedTemplate

from geometric_aware_sampling.experiments.base_experiment import BaseExperimentStrategy
from geometric_aware_sampling.experiments.geometric_aware_sampling.batch_observer_plugin import (
    BatchObserverPlugin,
)
from geometric_aware_sampling.experiments.geometric_aware_sampling.geometric_plugin import (
    GeometricPlugin,
)
from geometric_aware_sampling.experiments.geometric_aware_sampling.representation_plugin import (
    RepresentationPlugin,
)


class GeometricAwareSamplingStrategy(BaseExperimentStrategy):
    """

    Implements the Geometric Aware Sampling continual learning strategy
    as described in our paper

    """

    def create_cl_strategy(self):

        # ratio of new samples / buffer samples to use
        # setting replay_ration = 1.0 will double the batch size
        # for task_idx > 1, e.g. if batch_size is set to 64, we
        # have for task_idx > 1 a batch size of 128 with 64 new samples
        # and 64 buffer samples (replay samples)
        # this was the default before the introduction of the replay_ratio
        replay_ratio = 0.25
        assert (
            self.batch_size % (1.0 / replay_ratio) <= 1e-9
        ), "batch size must be divisible by 1 / replay_ratio"

        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GeometricPlugin(
                    # full cifar100 dataset (however this is just a theoretical value)
                    # we store less samples per task in the buffer (see q value inside the plugin)
                    mem_size=100000,
                    # batch size split for task_idx > 1 (replay samples versus new samples)
                    batch_size=(
                        int(self.batch_size * (1 - replay_ratio))
                        if replay_ratio < 1
                        else self.batch_size
                    ),
                    batch_size_mem=int(self.batch_size * replay_ratio),
                ),
                RepresentationPlugin(),
            ],
        )
