from avalanche.training.templates import SupervisedTemplate

from geometric_aware_sampling.experiments.base_experiment import BaseExperimentStrategy
from geometric_aware_sampling.experiments.goldilocks.goldilocks_plugin import (
    GoldilocksPlugin,
)


class GoldilocksBaselineStrategy(BaseExperimentStrategy):

    def create_cl_strategy(self):
        return SupervisedTemplate(
            **self.default_settings,
            plugins=[GoldilocksPlugin(mem_size=1000)],
        )
