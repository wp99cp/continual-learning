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

    experiments = [NaiveBaseline, ReplayBaseline]
    for i, experiment in enumerate(experiments):
        experiment = experiment(args)
        experiment.run()


if __name__ == "__main__":
    main()
