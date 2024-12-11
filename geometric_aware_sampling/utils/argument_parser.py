import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )

    parser.add_argument(
        "--res_path",
        type=str,
        default=None,
        help="Path to a results file to load. Skips the experiments and directly prints the results.",
    )

    return parser.parse_args()
