from typing import Any

import torch
from avalanche.benchmarks import AvalancheDataset
from avalanche.benchmarks.utils import DataAttribute
from avalanche.core import SupervisedPlugin, Template
from avalanche.training.templates import SupervisedTemplate


class LearningSpeedPlugin(SupervisedPlugin, supports_distributed=False):
    """
    A plugin that keeps track of the learning speed of the samples.
    """

    def __init__(self):
        super().__init__()

        self.classification_matrix = None
        self.epoch = 0

    def before_train_dataset_adaptation(self, strategy: "SupervisedTemplate", **kwargs):
        """Insert an additional field for the learning speed the dataset"""
        assert strategy.experience is not None

        # add the learning speed parameter to the dataset
        dataset: AvalancheDataset = strategy.experience.dataset
        strategy.experience.dataset = dataset.update_data_attribute(
            name="idx",
            new_value=DataAttribute(
                data=range(len(dataset)), name="idx", use_in_getitem=True
            ),
        )

    def before_training_exp(self, strategy: "SupervisedTemplate", **kwargs):

        print("DEBUG: reset the learning speed matrix")

        dataset: AvalancheDataset = strategy.experience.dataset

        # create a matrix of size strategy.train_epochs x len(dataset)
        self.classification_matrix = torch.zeros(
            strategy.train_epochs, len(dataset), dtype=torch.int8
        ).to(device=next(strategy.model.parameters()).device)

        self.epoch = 0  # reset the epoch counter

    def after_training_epoch(self, strategy: Template, *args, **kwargs) -> Any:
        self.epoch += 1

    def after_training_iteration(self, strategy: Template, *args, **kwargs) -> Any:
        """
        Update the classification matrix with the learning speed of the samples
        """

        # check if the samples are correctly classified
        # if so add one to the learning speed of the samples

        # get the index of the samples
        idxs = strategy.mbatch[-1]

        # correct classification
        output = strategy.mb_output  # of size (32, 10)
        target = strategy.mb_y

        # get the predicted class
        _, predicted = torch.max(output, 1)
        correct = (predicted == target).squeeze()

        # update the classification matrix
        self.classification_matrix[self.epoch, idxs] = correct.to(torch.int8)

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """
        Compute the learning speed of the samples
        """

        # compute the learning speed
        self.classification_matrix = self.classification_matrix.cpu()
        learning_speed = self.classification_matrix.float().mean(dim=0)

        print(f"Learning speed Histogram: {torch.histc(learning_speed, bins=10)}")
