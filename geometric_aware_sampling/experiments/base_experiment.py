import argparse
import datetime
from abc import abstractmethod

import torch
from avalanche.logging import TensorboardLogger, InteractiveLogger
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from geometric_aware_sampling.dataset.data_loader import load_dataset
from geometric_aware_sampling.evaluation.evaluation import get_evaluator
from geometric_aware_sampling.evaluation.model_tester_plugin import ModelTesterPlugin
from geometric_aware_sampling.models.model_loader import load_model
from geometric_aware_sampling.utils.hardware_info import print_hardware_info
from geometric_aware_sampling.utils.logging.tensor_board_logger import LogEnabledABC


def model_evaluation_callback(strategy, dataset, current_experience=None):
    """
    Callback to evaluate the model on the dataset after each experience

    We evaluate the model the classes of all experiences seen so far
    e.g., after the first experience, we evaluate on the first experience
    after the second experience, we evaluate on the first and second experience

    """

    if current_experience is None:
        assert (
            strategy.experience.current_experience is not None
        ), "No current experience"
        current_experience = strategy.experience.current_experience + 1

    strategy.eval(dataset.test_stream[:current_experience])


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
    ):
        """
        Initialize the experiment

        :param args: the command line arguments
        :param dataset_name: the name of the dataset, either "split_mnist" or "split_cifar100"
        :param model_name: the name of the model, currently only "slim_resnet18"
        """

        self.args = args
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.batch_size = batch_size
        self.train_epochs = train_epochs

        self.device = torch.device(
            f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
        )

        # set the default precision for matrix multiplication to high
        torch.set_float32_matmul_precision("high")

        # initialize the continual learning strategy and dataset
        self.cl_strategy = None
        self.cl_dataset = None
        self.model = None

        # setting if we should run an evaluation pass
        # after every epoch or after every experience/task
        self.run_evaluation_after_epoch = True

        method_name = self.__class__.__name__
        self.tensorboard_logger = TensorboardLogger(
            f"tb_data/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}__{method_name}"
        )

        self.__print_model_name()
        self.__setup__()

    def __setup__(self):
        print_hardware_info(self.args)

        ###################################
        # Load the dataset and model
        ###################################

        self.cl_dataset = load_dataset(
            self.dataset_name,
            print_summary=True,  # print summary statistics of the dataset / experience
            tensorboard_logger=self.tensorboard_logger,
        )

        self.model = load_model(
            model_name=self.model_name,
            cl_dataset=self.cl_dataset,
        )

        self.optimizer = Adam(
            self.model.parameters(), betas=(0.9, 0.999), lr=0.001, weight_decay=1e-5
        )

        self.criterion = CrossEntropyLoss()

        ###################################
        # Continual Learning Strategy
        ###################################

        self.eval_plugin = get_evaluator(
            self.cl_dataset.n_classes, [self.tensorboard_logger, InteractiveLogger()]
        )
        self.cl_strategy = self.create_cl_strategy()

        # add plugin to run evaluation after every epoch
        if self.run_evaluation_after_epoch:
            model_tester_plugin = ModelTesterPlugin(
                callback=model_evaluation_callback,
                dataset=self.cl_dataset,
            )
            self.cl_strategy.plugins.append(model_tester_plugin)

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

            self.cl_strategy.train(experience)

            if (
                not self.run_evaluation_after_epoch
            ):  # otherwise this is done by the plugin
                model_evaluation_callback(self.cl_strategy, self.cl_dataset, i)

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
        }
