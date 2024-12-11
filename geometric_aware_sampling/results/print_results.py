import numpy as np
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

    print("\n\n\n")
