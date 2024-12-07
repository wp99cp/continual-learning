import argparse

from geometric_aware_sampling.experiments.naive_baseline import NaiveBaseline
from geometric_aware_sampling.utils.hardware_info import print_hardware_info


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
    print_hardware_info(args)

    # #########################
    # Run the Experiment(s)
    # #########################

    baseline_naive = NaiveBaseline(args)
    baseline_naive.run()


if __name__ == "__main__":
    main()
