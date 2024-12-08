from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.problem_type import SupervisedProblem

from geometric_aware_sampling.experiments.base_experiment import BaseExperimentStrategy
from geometric_aware_sampling.experiments.goldilocks.goldilocks_plugin import (
    GoldilocksPlugin,
)


@property
def mb_task_id(self):
    """Current mini-batch task labels."""
    mbatch = self.mbatch
    assert mbatch is not None
    assert len(mbatch) >= 3, "Task label not found."

    # PATCH: the mb_task_id is the third element of the mbatch, not the last one
    # original: https://github.com/ContinualAI/avalanche/blob/625e46d9203878ed51457f6d7cd8d4e9fb05d093/avalanche/training/templates/problem_type/supervised_problem.py#L39
    return mbatch[2]


# TODO: verify that this is a bug and not a "feature" of the Avalanche library
#  if it's a bug we should open an issue in the Avalanche repository
# PATCH: override the mb_task_id property
SupervisedProblem.mb_task_id = mb_task_id


class GoldilocksBaselineStrategy(BaseExperimentStrategy):

    def create_cl_strategy(self):
        return SupervisedTemplate(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            train_epochs=10,
            train_mb_size=16,
            eval_mb_size=16,
            device=self.device,
            evaluator=self.eval_plugin,
            plugins=[GoldilocksPlugin(mem_size=200)],
        )
