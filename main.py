import argparse

from geometric_aware_sampling.experiments.geometric_aware_sampling.geometric_aware_sampling import (
    GeometricAwareSamplingStrategy,
)
from geometric_aware_sampling.experiments.goldilocks.goldilocks_experiment import (
    GoldilocksBaselineStrategy,
)
from geometric_aware_sampling.experiments.naive.naive_baseline import (
    NaiveBaselineStrategy,
)
from geometric_aware_sampling.experiments.replay.replay_baseline import (
    ReplayBaselineStrategy,
)
from geometric_aware_sampling.experiments.retrain_from_scratch.retrain_baseline import (
    RetrainBaselineStrategy,
)

from geometric_aware_sampling.utils.argument_parser import parse_arguments


def main():
    args = parse_arguments()

    ###################################
    # Experiment Settings and (global) Hyperparameters
    ###################################

    settings = {
        "args": args,
        "dataset_name": "split_cifar100",  # "split_cifar100", "split_mnist", "split_tiny_imagenet", or "split_fmnist
        "model_name": "slim_resnet18",  # "slim_resnet18"
        "batch_size": 16,  # for replay based strategies, the actual batch size is batch_size * 2
        "train_epochs": 10,
    }

    experiments = [
        GoldilocksBaselineStrategy,
        ReplayBaselineStrategy,
        NaiveBaselineStrategy,
        RetrainBaselineStrategy,
        GeometricAwareSamplingStrategy,
    ]
    for i, experiment in enumerate(experiments):
        experiment = experiment(**settings)
        experiment.run()


if __name__ == "__main__":
    main()
