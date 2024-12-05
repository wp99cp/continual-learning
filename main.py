import argparse

from geometric_aware_sampling.experiments import baseline
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
    # Start the experiment(s)
    # #########################

    # You can modify the following line to run different experiments.
    baseline.run(args)


if __name__ == "__main__":
    main()
