from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.problem_type import SupervisedProblem

from geometric_aware_sampling.experiments.base_experiment import BaseExperimentStrategy
from geometric_aware_sampling.experiments.goldilocks.goldilocks_plugin import (
    GoldilocksPlugin,
)


class GoldilocksBaselineStrategy(BaseExperimentStrategy):

    def create_cl_strategy(self):
        return SupervisedTemplate(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            train_epochs=2,
            train_mb_size=16,
            eval_mb_size=16,
            device=self.device,
            evaluator=self.eval_plugin,
            plugins=[GoldilocksPlugin(mem_size=200)],
        )
