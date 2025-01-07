import datetime

from torch.utils.tensorboard import SummaryWriter

from geometric_aware_sampling.experiments.geometric_aware_sampling.geometric_aware_sampling import (
    GeometricAwareSamplingStrategy,
)
from geometric_aware_sampling.experiments.goldilocks.goldilocks_experiment import (
    GoldilocksBaselineStrategy,
)
from geometric_aware_sampling.experiments.naive.naive_baseline import (
    NaiveBaselineStrategy,
)
from geometric_aware_sampling.experiments.ppp_loss.ppp_loss_experiment import (
    PPPLossStrategy,
)
from geometric_aware_sampling.experiments.replay.replay_baseline import (
    ReplayBaselineStrategy,
)
from geometric_aware_sampling.experiments.retrain_from_scratch.retrain_baseline import (
    RetrainBaselineStrategy,
)
from geometric_aware_sampling.experiments.run_experiments import run_experiments
from geometric_aware_sampling.results.print_results import print_results
from geometric_aware_sampling.utils.argument_parser import parse_arguments
from geometric_aware_sampling.utils.file_handler import (
    load_results_from_pkl,
    save_results_to_pkl,
)
from geometric_aware_sampling.utils.logging.settings import TENSORBOARD_DIR


def main():
    args = parse_arguments()

    ###################################
    # Experiment Settings and (global) Hyperparameters
    ###################################

    settings = {
        "args": args,
        "dataset_name": "split_cifar100",  # "split_cifar100", "split_mnist", "split_tiny_imagenet", or "split_fmnist
        "model_name": "slim_resnet18",  # "slim_resnet18", "resnet50", "resnet101", or "resnet152"
        "batch_size": 64,  # for replay based strategies, the actual batch size is batch_size * 2
        "train_epochs": 36,
    }

    # define the number of repetitions for each experiment
    # this is useful to get a more stable estimate of the performance
    repetitions = 1

    experiments = [
        # RetrainBaselineStrategy,
        # NaiveBaselineStrategy,
        # PPPLossStrategy,
        # ReplayBaselineStrategy,
        # GoldilocksBaselineStrategy,
        GeometricAwareSamplingStrategy,
    ]

    overall_results = {}

    ###################################
    # Run the Experiments and Collect Results
    ###################################

    if args.res_path is None:  # skip experiments if res_path is given
        overall_results = run_experiments(experiments, repetitions, settings)

    ###################################
    # Save Results to File
    ###################################

    now = datetime.datetime.now()
    path = f"{TENSORBOARD_DIR}/{now.strftime("%Y-%m-%d_%H-%M")}_results"

    if args.res_path is None:  # skip saving if res_path is given
        save_results_to_pkl(overall_results, path)

    ###################################
    # Load Results from File
    ###################################

    if args.res_path is not None:  # load results if res_path is given
        path = args.res_path
        overall_results = load_results_from_pkl(path)

    ###################################
    # Print Results and Create Figures
    ###################################

    writer = SummaryWriter(log_dir=path)
    print_results(overall_results, writer)


if __name__ == "__main__":
    main()
