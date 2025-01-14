import argparse
import datetime
from abc import abstractmethod

import torch
from avalanche.logging import TensorboardLogger, InteractiveLogger
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from geometric_aware_sampling.dataset.data_loader import load_dataset
from geometric_aware_sampling.evaluation.evaluation import get_evaluator
from geometric_aware_sampling.experiments.geometric_aware_sampling.batch_observer_plugin import (
    BatchObserverPlugin,
)
from geometric_aware_sampling.experiments.goldilocks.learning_speed_plugin import (
    SampleIdxPlugin,
)
from geometric_aware_sampling.models.model_loader import load_model
from geometric_aware_sampling.utils.hardware_info import print_hardware_info
from geometric_aware_sampling.utils.logged_lr_scheduler_plugin import (
    LoggedLRSchedulerPlugin,
)
from geometric_aware_sampling.utils.logging.settings import TENSORBOARD_DIR
from geometric_aware_sampling.utils.logging.tensor_board_logger import LogEnabledABC


class BaseExperimentStrategy(metaclass=LogEnabledABC):
    """

    Abstract Base class for all experiments

    """

    def __init__(
        self,
        args: argparse.Namespace,
        dataset_name: str = "split_mnist",
        model_name: str = "slim_resnet18",
        batch_size: int = 16,
        train_epochs: int = 5,
        seed: int = 42,
        n_experiences: int = 5,
        stop_after_n_experiences: int = -1,
    ):
        """
        Initialize the experiment

        :param args: the command line arguments
        :param dataset_name: the name of the dataset, either "split_mnist" or "split_cifar100"
        :param model_name: the name of the model, currently only "slim_resnet18"
        :param batch_size: the batch size
        :param train_epochs: the number of training epochs
        :param seed: the random seed
        :param n_experiences: the number of experiences
        :param stop_after_n_experiences: only train the first n_experiences, then stop
             -1 means train all experiences
        """

        self.args = args
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.seed = seed
        self.n_experiences = n_experiences
        self.stop_after_n_experiences = stop_after_n_experiences

        self.device = torch.device(
            f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
        )

        # set the default precision for matrix multiplication to high
        torch.set_float32_matmul_precision("high")

        # initialize the continual learning strategy and dataset
        self.cl_strategy = None
        self.cl_dataset = None
        self.model = None

        method_name = self.__class__.__name__
        self.tensorboard_logger = TensorboardLogger(
            f"{TENSORBOARD_DIR}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}__{method_name}"
        )
        self.report_base_settings()

        self.__print_model_name()
        self.__setup__()

    def report_base_settings(self):
        self.tensorboard_logger.writer.add_scalar(
            "hyperparameters/batch_size", self.batch_size
        )
        self.tensorboard_logger.writer.add_scalar(
            "hyperparameters/train_epochs", self.train_epochs
        )
        self.tensorboard_logger.writer.add_text("model_name", self.model_name)
        self.tensorboard_logger.writer.add_text("dataset_name", self.dataset_name)

    def __setup__(self):
        print_hardware_info(self.args)

        ###################################
        # Load the dataset and model
        ###################################

        self.cl_dataset = load_dataset(
            self.dataset_name,
            print_summary=True,  # print summary statistics of the dataset / experience
            n_experiences=self.n_experiences,
            seed=self.seed,
            tensorboard_logger=self.tensorboard_logger,
        )

        self.model = load_model(
            model_name=self.model_name,
            cl_dataset=self.cl_dataset,
        )

        self.optimizer = SGD(
            self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005
        )

        # learning rate scheduler (could also be ReduceLROnPlateau or
        # any other scheduler from torch.optim.lr_scheduler)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.train_epochs)

        self.criterion = CrossEntropyLoss()

        ###################################
        # Continual Learning Strategy
        ###################################

        self.eval_plugin = get_evaluator(
            self.cl_dataset.n_classes, [self.tensorboard_logger, InteractiveLogger()]
        )
        self.cl_strategy = self.create_cl_strategy()

        # add learning rate scheduler
        lr_scheduler_plugin = LoggedLRSchedulerPlugin(
            scheduler=self.scheduler,
            # the following argument is only used for the ReduceLROnPlateau scheduler
            # not used for CosineAnnealingLR
            # metric="train_loss",  # we should not use validation loss as this leaks information
        )
        self.cl_strategy.plugins.append(lr_scheduler_plugin)

        # add batch observer plugin

        sample_idx_plugin = SampleIdxPlugin()
        self.cl_strategy.plugins.append(sample_idx_plugin)

        batch_observer_plugin = BatchObserverPlugin(
            normalize_steps=True, strategy_name=self.__class__.__name__
        )
        self.cl_strategy.plugins.append(batch_observer_plugin)

    def __print_model_name(self):
        print(
            f"""

################################################
################################################
#
#   {self.__class__.__name__}
#
################################################
################################################

                    """
        )

    @abstractmethod
    def create_cl_strategy(self):
        """
        Create the continual learning strategy.

        This method must be implemented by the subclass.
        You may access the mode, dataset, optimizer, criterion, and device
        from the instance variables of the Experiment class.

        :return: the continual learning strategy
        """
        raise NotImplementedError

    def run(self):
        """
        Run the experiment using the specified continual learning strategy
        """

        print("\n\n####################\nStarting experiment\n####################\n\n")
        print(f"Using {self.cl_strategy.__class__.__name__} strategy\n")

        for i, experience in enumerate(self.cl_dataset.train_stream, 1):
            print(
                f"\n - Experience {i} / {self.cl_dataset.n_experiences} with classes {experience.classes_in_this_experience}"
            )

            if i > 1:
                # set the learning rate
                self.optimizer.param_groups[0]["lr"] = 0.05

            self.cl_strategy.train(
                experience,
                eval_streams=[self.cl_dataset.test_stream[:i]],
                tensorboard_logger=self.tensorboard_logger,
            )

            if i == self.stop_after_n_experiences:
                print(
                    f"Stop training after {self.stop_after_n_experiences} experiences!"
                )
                break

        print("\n\n####################\nExperiment finished\n####################\n\n")

    @property
    def default_settings(self):
        return {
            "model": self.model,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
            "train_epochs": self.train_epochs,
            "train_mb_size": self.batch_size,
            "eval_mb_size": self.batch_size,
            "device": self.device,
            "evaluator": self.eval_plugin,
            "eval_every": 1,  # eval after every epoch
        }

    def get_results(self):
        """
        Used to return a dict of the experiment results
        :return: dict
        """

        # extract the results form the evaluation plugin
        plugins = self.cl_strategy.plugins
        eval_plugin = [
            p for p in plugins if p.__class__.__name__ == "EvaluationPlugin"
        ][0]

        return eval_plugin.all_metric_results
