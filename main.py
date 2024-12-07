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

    # #########################
    # Run the Experiment(s)
    # #########################

    experiments = [
        NaiveBaseline,
        ReplayBaseline,
    ]

    for i, experiment in enumerate(experiments):

        print(
            f"""
            
################################################
################################################
#
#   Running {experiment.__name__}... ({i + 1}/{len(experiments)})
#
################################################
################################################

            """
        )
        experiment = experiment(args)
        experiment.run()


if __name__ == "__main__":
    main()
