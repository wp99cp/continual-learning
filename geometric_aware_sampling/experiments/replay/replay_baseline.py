from avalanche.training import Replay

from geometric_aware_sampling.experiments.base_experiment import BaseExperimentStrategy


class ReplayBaselineStrategy(BaseExperimentStrategy):
    """

    Baseline experiment using the Naive continual learning strategy
    from the Avalanche library.

    """

    def create_cl_strategy(self):
        return Replay(**self.default_settings)
