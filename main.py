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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    settings = {
        "args": args,
        "dataset_name": "split_cifar100",  # "split_cifar100" or "split_mnist"
        "model_name": "slim_resnet18",  # "slim_resnet18"
        "batch_size": 16,  # for replay based strategies, the actual batch size is batch_size * 2
        "train_epochs": 10,
    }

    experiments = [
        GoldilocksBaselineStrategy,
        ReplayBaselineStrategy,
        NaiveBaselineStrategy,
        GeometricAwareSamplingStrategy,
    ]
    for i, experiment in enumerate(experiments):
        experiment = experiment(**settings)
        experiment.run()


if __name__ == "__main__":
    main()
