from avalanche.training.templates import SupervisedTemplate

from geometric_aware_sampling.experiments.base_experiment import BaseExperimentStrategy
from geometric_aware_sampling.experiments.geometric_aware_sampling.geometric_plugin import (
    GeometricPlugin,
)
from geometric_aware_sampling.experiments.geometric_aware_sampling.representation_plugin import (
    RepresentationPlugin
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
                    mem_size=1000  # ~ 10% of the cifar100 dataset per task
                ),
                RepresentationPlugin()
            ],
        )
