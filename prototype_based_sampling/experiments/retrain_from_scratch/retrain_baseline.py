from avalanche.training import FromScratchTraining

from prototype_based_sampling.experiments.base_experiment import BaseExperimentStrategy


class RetrainBaselineStrategy(BaseExperimentStrategy):
    """

    Baseline experiment using the Naive continual learning strategy
    from the Avalanche library.

    """

    def create_cl_strategy(self):
        return FromScratchTraining(**self.default_settings)
