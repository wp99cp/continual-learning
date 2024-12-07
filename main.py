import argparse

from geometric_aware_sampling.experiments.naive_baseline import NaiveBaseline
from geometric_aware_sampling.experiments.replay_baseline import ReplayBaseline


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
    }

    experiments = [NaiveBaseline, ReplayBaseline]
    for i, experiment in enumerate(experiments):
        experiment = experiment(**settings)
        experiment.run()


if __name__ == "__main__":
    main()
