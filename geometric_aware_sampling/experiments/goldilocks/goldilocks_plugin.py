from typing import Optional, Any

import torch
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.core import Template
from avalanche.training.plugins import (
    SupervisedPlugin,
)
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
)
from avalanche.training.templates import SupervisedTemplate
from packaging.version import parse

from geometric_aware_sampling.experiments.goldilocks.learning_speed_plugin import (
    LearningSpeedPlugin,
)


#
# Documentation about avalanche library
# --> see https://avalanche.continualai.org/from-zero-to-hero-tutorial/04_training#training-and-evaluation-loops
# and https://avalanche-api.continualai.org/en/v0.6.0/generated/avalanche.core.SupervisedPlugin.html
#


class GoldilocksPlugin(SupervisedPlugin, supports_distributed=False):
    """
    Experience replay plugin based in the Goldilocks buffer sampling strategy.
    See https://arxiv.org/abs/2406.09935 for more details.

    The `before_training_exp` callback is implemented in order to use the
    dataloader that creates mini-batches with examples from both training
    data and external memory. The examples in the mini-batch is balanced
    such that there are the same number of examples for each experience.

    The `after_training_exp` callback is implemented in order to add new
    patterns to the external memory based on the learning speed of the samples.
    Especially, we add samples that with a learning speed between the qth and
    s-th quantile of the learning speed distribution.

    In order to track the learning speed of the samples, we use the `LearningSpeedPlugin`
    plugin which adds a dimension to the data that contains the learning speed of the samples.
    The learning speed is updated on every mini-batch.

    The :mem_size: attribute controls the total number of samples to be stored
    in the external memory.

    :param batch_size: the size of the data batch. If set to `None`, it
        will be set equal to the strategy's batch size.
    :param batch_size_mem: the size of the memory batch. If
        `task_balanced_dataloader` is set to True, it must be greater than or
        equal to the number of tasks. If its value is set to `None`
        (the default value), it will be automatically set equal to the
        data batch size.
    :param task_balanced_dataloader: if True, buffer data loaders will be
            task-balanced, otherwise it will create a single dataloader for the
            buffer samples.
    :param storage_policy: The policy that controls how to add new exemplars
                           in memory

    Based on the implementation of the ReplayPlugin from the Avalanche library.
    Release under the MIT License. Source:
    https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/plugins/replay.py

    """

    def __init__(
        self,
        mem_size: int = 200,
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
    ):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        self.has_added_learning_speed_plugin = False

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )

    def before_training(self, strategy: Template, *args, **kwargs) -> Any:
        """Adds the learning speed plugin to the strategy."""

        if self.has_added_learning_speed_plugin:
            return  # we need to add the LearningSpeedPlugin only once

        self.has_added_learning_speed_plugin = True

        if not any(isinstance(p, LearningSpeedPlugin) for p in strategy.plugins):
            strategy.plugins.append(LearningSpeedPlugin())
        else:
            print("WARNING: LearningSpeedPlugin already added to the strategy")

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        """
        before_training_exp to customize the dataloader

        Dataloader to build batches containing examples from both memories and
        the training dataset
        """

        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = strategy.train_mb_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size

        assert strategy.adapted_dataset is not None

        other_dataloader_args = dict()

        if "ffcv_args" in kwargs:
            other_dataloader_args["ffcv_args"] = kwargs["ffcv_args"]

        if "persistent_workers" in kwargs:
            if parse(torch.__version__) >= parse("1.7.0"):
                other_dataloader_args["persistent_workers"] = kwargs[
                    "persistent_workers"
                ]

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            **other_dataloader_args
        )

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """
        uses after_training_exp to update the buffer after each training experience
        """

        # TODO: add samples based on the learning speed
        self.storage_policy.update(strategy, **kwargs)
