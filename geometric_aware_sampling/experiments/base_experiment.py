import argparse
from abc import ABC, abstractmethod
from datetime import datetime

import torch
from avalanche.logging import TensorboardLogger
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from geometric_aware_sampling.dataset.data_loader import load_dataset
from geometric_aware_sampling.evaluation.evaluation import get_evaluator
from geometric_aware_sampling.models.model_loader import load_model


class BaseExperiment(ABC):
    """

    Abstract Base class for all experiments

    """

    def __init__(
        self,
        args: argparse.Namespace,
        dataset_name: str = "split_mnist",
        model_name: str = "slim_resnet18",
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

        self.device = torch.device(
            f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
        )

        # set the default precision for matrix multiplication to high
        torch.set_float32_matmul_precision("high")

        # initialize the continual learning strategy and dataset
        self.cl_strategy = None
        self.cl_dataset = None
        self.model = None

        # define the tensorboard logger
        self.tensorboard_logger = TensorboardLogger(f"tb_data/{datetime.now()}")

        self.__setup__()

    def __setup__(self):
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
            self.cl_dataset.n_classes, [self.tensorboard_logger]
        )
        self.cl_strategy = self.create_cl_strategy()

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

            # we evaluate the model the classes of all experiences seen so far
            # e.g., after the first experience, we evaluate on the first experience
            # after the second experience, we evaluate on the first and second experience
            self.cl_strategy.eval(self.cl_dataset.test_stream[:i])

        print("\n\n####################\nExperiment finished\n####################\n\n")
