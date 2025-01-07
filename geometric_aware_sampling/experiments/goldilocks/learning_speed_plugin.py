from typing import Any

import torch
from avalanche.benchmarks import AvalancheDataset
from avalanche.benchmarks.utils import DataAttribute
from avalanche.core import SupervisedPlugin, Template
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.problem_type import SupervisedProblem


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


class SampleIdxPlugin(SupervisedPlugin, supports_distributed=False):

    def before_train_dataset_adaptation(self, strategy: "SupervisedTemplate", **kwargs):
        """
        In order to keep track of the learning speed of a sample we need to inject a
        sample index in the dataset. This index will be used to update the learning speed
        of the samples and is included in the dataset as a new data attribute.

        In order to correctly report the mb_task_id, we need to override the mb_task_id property
        of the SupervisedProblem class. This is a patch to fix a bug in the Avalanche library,
        where the mb_task_id is not correctly reported when additional data attributes are added
        to the dataset.

        """
        assert strategy.experience is not None

        # add the learning speed parameter to the dataset
        dataset: AvalancheDataset = strategy.experience.dataset
        strategy.experience.dataset = dataset.update_data_attribute(
            name="idx",
            new_value=DataAttribute(
                data=range(len(dataset)), name="idx", use_in_getitem=True
            ),
        )


class LearningSpeedPlugin(SupervisedPlugin, supports_distributed=False):
    """
    A plugin that keeps track of the learning speed of the samples of the current task.

    The learning speed is computed as the ratio of the number of times a sample is correctly classified
    over the number of epochs it is trained on (under the assumption that every sample is included in exactly
    one mini-batch per epoch).

    We only keep track of the learning speed of the samples from the current task. Samples injected
    by the replay strategy are not considered.

    """

    def __init__(self):
        super().__init__()

        self.__correctly_classified_counter = None
        self.learning_speed = None
        self.epoch_counter = 0

    def before_train_dataset_adaptation(self, strategy: "SupervisedTemplate", **kwargs):

        self.learning_speed = torch.zeros(
            len(strategy.experience.dataset), dtype=torch.float32
        )

    def before_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """
        Initialize the learning speed counter, for every task we reset the counter.
        """

        device = next(strategy.model.parameters()).device
        self.__correctly_classified_counter = torch.zeros(
            len(strategy.experience.dataset), dtype=torch.int8
        ).to(device)

        # reset the epoch counter
        self.epoch_counter = 0

    def after_training_iteration(self, strategy: Template, *args, **kwargs) -> Any:
        """
        Track if the samples are classified correctly or not
        """

        # get the index of the samples
        assert strategy.mbatch is not None, "Mini-batch not found"
        assert len(strategy.mbatch) == 4, "index not found"

        indexes = strategy.mbatch[3]

        # track the correct classified samples
        _, predicted = torch.max(strategy.mb_output, 1)
        correct = (predicted == strategy.mb_y).squeeze().to(torch.int8)

        # filter for current task
        current_task_id = strategy.experience.current_experience
        indexes = indexes[strategy.mb_task_id == current_task_id]
        correct = correct[strategy.mb_task_id == current_task_id]

        # update the learning speed counter
        self.__correctly_classified_counter[indexes] += correct

    def after_training_epoch(self, strategy: "SupervisedTemplate", **kwargs):
        """
        Compute the learning speed of the samples
        """

        self.epoch_counter += 1

        # we only compute the learning speed after the training phase,
        # but we need to do it before the after_training_exp method
        # such that the result can be used in other plugins
        if self.epoch_counter < strategy.train_epochs:
            return

        # compute the learning speed
        self.__correctly_classified_counter = self.__correctly_classified_counter.cpu()
        self.learning_speed = (
            self.__correctly_classified_counter.float() / self.epoch_counter
        )

        print(
            f"""
learning_speed: μ={self.learning_speed.mean():.4f} σ={self.learning_speed.std():.4f}
learning_speed hist: {torch.histc(self.learning_speed, bins=10, min=0, max=1).numpy()}
"""
        )
