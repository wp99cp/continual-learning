from avalanche.training import BalancedExemplarsBuffer
from avalanche.training.templates import SupervisedTemplate

from geometric_aware_sampling.experiments.base_experiment import BaseExperimentStrategy
from geometric_aware_sampling.experiments.goldilocks.goldilocks_plugin import (
    GoldilocksPlugin,
)


class GeometricAwareSampling(BalancedExemplarsBuffer):
    """
    Implements our geometric aware sampling strategy from the external replay buffer.
    """

    # TODO: here we implement our main code of the project

    pass


class GeometricAwareSamplingStrategy(BaseExperimentStrategy):
    """

    Implements the Geometric Aware Sampling continual learning strategy
    as described in our paper

    """

    def create_cl_strategy(self):
        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                GoldilocksPlugin(storage_policy=GeometricAwareSampling(200)),
            ],
        )
