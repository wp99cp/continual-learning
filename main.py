import datetime

from torch.utils.tensorboard import SummaryWriter

from prototype_based_sampling.experiments.prototype_based_replay_selection.prototype_based_replay_selection import (
    Baseline_Goldilocks_Random,
    Baseline_Random_Random,
    Baseline_Icarl_Random,
    PrototypeBased_Goldilocks_MaxScatter,
    PrototypeBased_Goldilocks_InvertedDistance,
    PrototypeBased_Goldilocks_ExponentialDistance,
)
from prototype_based_sampling.experiments.run_experiments import run_experiments
from prototype_based_sampling.results.print_results import print_results
from prototype_based_sampling.utils.argument_parser import parse_arguments
from prototype_based_sampling.utils.file_handler import (
    load_results_from_pkl,
    save_results_to_pkl,
)
from prototype_based_sampling.utils.logging.settings import TENSORBOARD_DIR


def main():
    args = parse_arguments()

    ###################################
    # Experiment Settings and
    # (global) Hyperparameters
    ###################################

    # if enabled we randomize the class-task mapping for every repetition
    # this is useful to get a more stable estimate of the performance
    randomize_class_task_mapping = False

    settings = {
        "args": args,
        "dataset_name": "split_cifar100",  # "split_cifar100", "split_mnist", "split_tiny_imagenet", or "split_fmnist
        "model_name": "slim_resnet18",  # "slim_resnet18", "resnet50", "resnet101", or "resnet152"
        "n_experiences": 5,  # thus we have the same setup as for the GoldiLockPaper
        "stop_after_n_experiences": 3,  # only trains the first n_experiences resp. tasks then stops
        "batch_size": 64,  # for replay based strategies, the actual batch size is batch_size * 2
        "train_epochs": 100,
    }

    # define the number of repetitions for each experiment
    # this is useful to get a more stable estimate of the performance
    repetitions = 5

    experiments = [
        # base baselines without a ER
        #
        # RetrainBaselineStrategy,
        # NaiveBaselineStrategy,
        #
        #
        # experiments used for the report
        Baseline_Random_Random,
        Baseline_Goldilocks_Random,
        Baseline_Icarl_Random,
        PrototypeBased_Goldilocks_MaxScatter,
        PrototypeBased_Goldilocks_InvertedDistance,
        PrototypeBased_Goldilocks_ExponentialDistance,
    ]

    overall_results = {}

    ###################################
    # Run the Experiments and Collect Results
    ###################################

    # skip experiments if res_path is given
    if args.res_path is None:
        overall_results = run_experiments(
            experiments, repetitions, settings, randomize_class_task_mapping
        )

    ###################################
    # Save Results to File
    ###################################

    now = datetime.datetime.now()
    path = f"{TENSORBOARD_DIR}/{now.strftime('%Y-%m-%d_%H-%M')}_results"

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
