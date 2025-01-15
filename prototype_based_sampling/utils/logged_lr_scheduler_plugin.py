from typing import Any

from avalanche.core import Template
from avalanche.training.plugins import LRSchedulerPlugin


class LoggedLRSchedulerPlugin(LRSchedulerPlugin):
    """
    Extends the LRSchedulerPlugin to log the learning rate and patience to tensorboard
    before each training epoch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 0

    def before_training_exp(self, strategy, *args, **kwargs):
        self.epoch = 0  # reset epoch counter

    def before_training_epoch(self, strategy: Template, *args, **kwargs) -> Any:
        experience = strategy.experience.current_experience
        train_epochs = strategy.train_epochs
        kwargs["tensorboard_logger"].writer.add_scalar(
            f"lr_scheduler/lr/Task{str(experience).zfill(3)}",
            self.scheduler.get_last_lr()[0],
            self.epoch,
        )

        kwargs["tensorboard_logger"].writer.add_scalar(
            "lr_scheduler/lr",
            self.scheduler.get_last_lr()[0],
            experience * train_epochs + self.epoch,
        )

        # check if self.scheduler is ReduceLROnPlateau
        if hasattr(self.scheduler, "num_bad_epochs") and hasattr(
            self.scheduler, "patience"
        ):
            kwargs["tensorboard_logger"].writer.add_scalar(
                f"lr_scheduler/remaining_patience/Task{str(experience).zfill(3)}",
                self.scheduler.patience - self.scheduler.num_bad_epochs,
                self.epoch,
            )

            kwargs["tensorboard_logger"].writer.add_scalar(
                "lr_scheduler/remaining_patience",
                self.scheduler.patience - self.scheduler.num_bad_epochs,
                experience * train_epochs + self.epoch,
            )

        self.epoch += 1
