import math

import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.patches import Patch
from torch.utils.tensorboard import SummaryWriter

from geometric_aware_sampling.results.evolution_graphs.top1_accuracy_evolution_graph import (
    top1_accuracy_evolution_graph,
)


def print_results(overall_results, writer: SummaryWriter):

    print("\n\n\n")
    for strategy_name, results in overall_results.items():
        top1_accuracy_evolution_graph(writer, strategy_name, results)

        # write mean and std of the final Top1 Accuracy to tensorboard
        metrics_name = "Top1_Acc_Stream/eval_phase/test_stream/Task004"
        top1_accuracies = [res[metrics_name] for res in results]

        top1_accuracies = [res[1] for res in top1_accuracies]
        mean_final_top1_acc = np.array(top1_accuracies[:][-1]).mean()
        std_final_top1_accu = np.array(top1_accuracies[:][-1]).std()
        writer.add_scalar(f"{strategy_name}/mean_Top1_Accuracy", mean_final_top1_acc)
        writer.add_scalar(f"{strategy_name}/std_Top1_Accuracy", std_final_top1_accu)

        __strategy_name = strategy_name.split("__")[-1]

        ########################################################################################################
        # START: accuracy history plot (only for first task)
        ########################################################################################################

        # TODO the following only consider the first run

        # class accuracy history plot (only for first task)
        class_acc_history = []

        for key in results[0].keys():
            if "Top1_Acc_Exp/eval_phase/test_stream" in key:

                np_array = np.array(results[0][key][1])
                class_acc_history.append(np_array)

        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

        for i, class_acc in enumerate(class_acc_history):
            xs = np.linspace(i, 5, len(class_acc))
            ax.plot(xs, class_acc, label=f"Testset with classes of task {i}")

        ax.set(
            xlabel=f"Start Training of Task i",
            ylabel="Accuracy",
            title=f"Accuracy History {__strategy_name}",
        )
        ax.legend()

        for i in range(1, 5):
            ax.axvline(x=i, color="k", linestyle="--", alpha=0.25)

        # save the plot to tensorboard
        writer.add_figure(
            f"{__strategy_name}/class_accuracy_history", fig, global_step=0
        )

        ########################################################################################################
        # END: accuracy history plot (only for first task)
        ########################################################################################################

        ########################################################################################################
        # START: forgetting plot
        ########################################################################################################

        # TODO the following only consider the first run

        # class accuracy history plot (only for first task)
        class_acc_history = []

        for key in results[0].keys():
            if "Top1_Acc_Exp/eval_phase/test_stream" in key:
                np_array = np.array(results[0][key][1])
                class_acc_history.append(np_array)

        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

        for i, class_acc in enumerate(class_acc_history):
            xs = np.linspace(i, 5, len(class_acc))

            last_train_idx = math.floor(float(len(class_acc)) / (5 - i))
            train_acc = np.max(class_acc[:last_train_idx])
            forgetting = train_acc - class_acc

            # set forgetting to zero before learning on the experience
            forgetting[:last_train_idx] = 0
            ax.plot(xs, forgetting, label=f"Testset with classes of task {i}")

        ax.set(
            xlabel=f"Start Training of Task i",
            ylabel="Forgetting",
            title=f"Forgetting {__strategy_name}",
        )

        ax.legend()

        for i in range(1, 5):
            ax.axvline(x=i, color="k", linestyle="--", alpha=0.25)

        # show the plot

        # save the plot to tensorboard
        writer.add_figure(f"{__strategy_name}/forgetting", fig, global_step=0)

        ########################################################################################################
        # END: forgetting plot
        ########################################################################################################

        ########################################################################################################
        # START: correctly classified samples plot
        ########################################################################################################

        # TODO the following only consider the first run

        correctly_classified = []
        length = -1

        for key in results[0].keys():
            if "ClassificationTracker_Exp/eval_phase/test_stream/Task000/Exp000" in key:
                correctly_classified.append(np.array(results[0][key][1]))

                if length == -1:
                    length = len(results[0][key][1])
                else:
                    assert length == len(results[0][key][1])

        last_train_idx = math.floor(length / 5.0)

        # sort correctly_classified by the accumulated number of correctly classified samples
        # over the range [0, last_train_idx]
        correctly_classified = sorted(
            correctly_classified, key=lambda x: np.sum(x[:last_train_idx])
        )

        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

        cmap = cm.get_cmap("viridis")  # Get the reversed colormap
        ax.imshow(np.array(correctly_classified), cmap=cmap, aspect="auto")

        # use special labels [0, 1, 2, 3, 4, 5] for the x-axis
        # evenly spaced on the range [0, length]
        ax.set_xticks(np.linspace(0, length, 6))
        ax.set_xticklabels([0, 1, 2, 3, 4, 5])

        # hide y-axis
        ax.get_yaxis().set_visible(False)

        ax.set(
            xlabel="Start of Training Task i",
            title=f"Correctly Classified Samples {__strategy_name}",
        )

        # Extract colors from the colormap for 0 and 1
        color_incorrect = cmap(0.0)
        color_correct = cmap(1.0)

        # Define legend manually using exact colors
        legend_labels = [
            Patch(color=color_incorrect, label="Incorrect"),
            Patch(color=color_correct, label="Correct"),
        ]
        ax.legend(handles=legend_labels, loc="upper right")

        # save the plot to tensorboard
        writer.add_figure(
            f"{__strategy_name}/correctly_classified_samples", fig, global_step=0
        )

        ########################################################################################################
        # END: correctly classified samples plot
        ########################################################################################################

    print("\n\n\n")
