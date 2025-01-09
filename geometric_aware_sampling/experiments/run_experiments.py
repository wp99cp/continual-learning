import datetime

from geometric_aware_sampling.utils.file_handler import save_results_to_pkl
from geometric_aware_sampling.utils.logging.settings import TENSORBOARD_DIR


def run_experiments(
    experiments, repetitions, settings, randomize_class_task_mapping: bool
):
    """
    Run the experiments and return the results
    """

    overall_results = {}

    has_been_interrupted = False
    for rep in range(repetitions):
        if has_been_interrupted:
            break

        for i, experiment_class in enumerate(experiments):

            exp_name = f"{i}__{experiment_class.__name__}"

            if exp_name not in overall_results:
                overall_results[exp_name] = []

            try:
                seed = rep * 42 if randomize_class_task_mapping else 42
                exp = experiment_class(seed=seed, **settings)
                exp.run()
                overall_results[exp_name].append(exp.get_results())

            # catch keyboard interrupt to stop the experiments
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting remaining repetitions.")

                # save the collected results and exit
                has_been_interrupted = True
                break

            # continue with other experiments if one fails
            except Exception as e:
                print(f"Error in experiment {exp_name}: {e}")

        now = datetime.datetime.now()
        path = f"{TENSORBOARD_DIR}/{now.strftime("%Y-%m-%d_%H-%M")}_results_rep-{rep}"
        save_results_to_pkl(overall_results, path)

    return overall_results
