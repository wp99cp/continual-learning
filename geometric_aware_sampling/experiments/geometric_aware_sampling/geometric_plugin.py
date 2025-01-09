from typing import Any, Type

import torch
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.core import Template
from avalanche.training.plugins import (
    SupervisedPlugin,
)
from avalanche.training.templates import SupervisedTemplate
from packaging.version import parse

from geometric_aware_sampling.experiments.geometric_aware_sampling.geometric_balanced_buffer import (
    GeometricBalancedBuffer,
)
from geometric_aware_sampling.experiments.geometric_aware_sampling.geometric_sampling_strategy import (
    BufferSamplingStrategy,
)
from geometric_aware_sampling.experiments.goldilocks.learning_speed_plugin import (
    LearningSpeedPlugin,
)


class GeometricPlugin(SupervisedPlugin, supports_distributed=False):
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

    :param replay_ratio: the ratio of replay samples to new samples in the mini-batch.
    :param mem_size: attribute controls the total number of samples to be stored
        in the external memory.
    :param task_balanced_dataloader: if True, buffer data loaders will be
        task-balanced, otherwise it will create a single dataloader for the
        buffer samples.
    :param upper_quantile: the upper quantile of the learning speed distribution that will
        never be included in the buffer
    :param lower_quantile: the lower quantile of the learning speed distribution that will
        never be included in the buffer
    :param q: ratio of training samples to keep, sampled using Goldilocks
    :param p: ratio of buffer samples to use 1.0 means that we can use all samples, as long as the replay_ratio
        is below q, this means that we replay a sample at most 1 time per epoch. If p is above 1.0, we treat
        p as a fixed value for the number of samples to replay per epoch.
    :param sample_per_epoch: if True, the plugin will sample new replay samples
        at the beginning of each epoch. Otherwise, it will sample new replay samples
        at the beginning of each experience.

    Based on the implementation of the ReplayPlugin from the Avalanche library.
    Release under the MIT License. Source:
    https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/plugins/replay.py

    """

    def __init__(
        self,
        sampling_strategy: Type[BufferSamplingStrategy],
        replay_ratio: float = 0.25,
        mem_size: int = 200,
        task_balanced_dataloader: bool = False,
        upper_quantile: float = 1 - 0.44,  # chosen according to the paper, figure 4 (b)
        lower_quantile: float = 0.12,  # chosen according to the paper, figure 4 (b)
        q: float = 0.4,
        p: float = 1.0,
        sample_per_epoch: bool = True,  # we want new samples per epoch
    ):
        super().__init__()
        self.batch_size = None
        self.batch_size_mem = None
        self.mem_size = mem_size
        self.replay_ratio = replay_ratio
        self.task_balanced_dataloader = task_balanced_dataloader

        self.has_added_learning_speed_plugin = False

        # The storage policy samples the data based on the learning speed
        # and stores the samples in the external memory.
        self.storage_policy = GeometricBalancedBuffer(
            max_size=self.mem_size,
            adaptive_size=True,
            upper_quantile_ls=upper_quantile,
            lower_quantile_ls=lower_quantile,
            q=q,
            p=p,
            sampling_strategy=sampling_strategy,
        )

        self.sample_per_epoch = sample_per_epoch
        self.task_idx = 0

    def before_training(self, strategy: Template, *args, **kwargs) -> Any:
        """Adds the learning speed plugin to the strategy."""

        if self.has_added_learning_speed_plugin:
            return  # we need to add the LearningSpeedPlugin only once

        self.has_added_learning_speed_plugin = True

        if not any(isinstance(p, LearningSpeedPlugin) for p in strategy.plugins):
            strategy.plugins.append(LearningSpeedPlugin())
        else:
            print("WARNING: LearningSpeedPlugin already added to the strategy")

    def __sample_new_replay_samples(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):

        # save buffer as local var to only call it once
        buffer = self.storage_policy.get_buffer(
            current_model=strategy.model, experience_dataset=strategy.experience.dataset
        )

        if len(buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

            # batch size split for task_idx > 1 (replay samples versus new samples)
        global_batch_size = strategy.train_mb_size

        assert (
            global_batch_size % (1.0 / self.replay_ratio) <= 1e-9
        ), "batch size must be divisible by (1 / replay_ratio)"

        batch_size = (
            int(global_batch_size * (1 - self.replay_ratio))
            if self.replay_ratio < 1
            else global_batch_size
        )
        batch_size_mem = int(global_batch_size * self.replay_ratio)

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
            buffer,
            oversample_small_tasks=True,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            **other_dataloader_args
        )

    def before_training_epoch(self, strategy: "SupervisedTemplate", **kwargs):

        if self.sample_per_epoch:
            self.__sample_new_replay_samples(strategy, **kwargs)

    def before_training_exp(self, strategy: "SupervisedTemplate", **kwargs):

        if not self.sample_per_epoch:
            self.__sample_new_replay_samples(strategy, **kwargs)

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """
        uses after_training_exp to update the buffer after each training experience
        """

        self.task_idx += 1
        self.storage_policy.post_adapt(strategy, strategy.experience)
        if hasattr(self.storage_policy, "log_buffer_summary") and callable(
            getattr(self.storage_policy, "log_buffer_summary")
        ):
            self.storage_policy.log_buffer_summary(kwargs, self.task_idx)
