from avalanche.training import Naive

from geometric_aware_sampling.experiments.base_experiment import BaseExperimentStrategy
from geometric_aware_sampling.experiments.naive.forgetting_eval_plugin import (
    ForgettingEvalPlugin,
)


class NaiveBaselineStrategy(BaseExperimentStrategy):
    """

    Baseline experiment using the Naive continual learning strategy
    from the Avalanche library.

    """

    def create_cl_strategy(self):
        return Naive(**self.default_settings, plugins=[ForgettingEvalPlugin()])
