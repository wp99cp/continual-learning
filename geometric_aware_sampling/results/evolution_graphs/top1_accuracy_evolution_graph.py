from torch.utils.tensorboard import SummaryWriter

from geometric_aware_sampling.results.evolution_graphs.evolution_graph import (
    get_epoch_evolution_graph,
)


def top1_accuracy_evolution_graph(writer: SummaryWriter, strategy_name, results):
    """

    This figure shows the evolution of the Top1 Accuracy over the epochs.

    The x-axis represents the epoch, the y-axis the Top1 Accuracy, and the z-axis describes
    the test set on which the model was evaluated. E.g. for task 3, the model was evaluated on
    the test set including samples from task 1, 2, and 3 in equal proportions.

    Most important is therefor the line for the last task, thus the chosen color map.

    """

    metrics_name = "Top1_Acc_Stream/eval_phase/test_stream"
    res_dict = {}
    for task_idx in range(5):
        task_name = f"Task{task_idx:03d}"
        full_metric_name = f"{metrics_name}/{task_name}"

        task_results = [res[full_metric_name] for res in results]
        task_results = [res[1] for res in task_results]  # throw away the epoch idx

        res_dict[task_idx] = task_results

    fig = get_epoch_evolution_graph(
        fig_title=f"{strategy_name} - Top1 Accuracy",
        x_label="Epoch",
        y_label="Top1 Accuracy",
        z_label="Tested on Tasks 1, ..., x",
        data=res_dict,
        with_std=False,
    )
    writer.add_figure(f"{strategy_name}/Top1_Accuracy", fig)

    fig = get_epoch_evolution_graph(
        fig_title=f"{strategy_name} - Top1 Accuracy",
        x_label="Epoch",
        y_label="Top1 Accuracy",
        z_label="Tested on Tasks 1, ..., x",
        data=res_dict,
        with_std=True,
    )
    writer.add_figure(f"{strategy_name}/Top1_Accuracy_with_std", fig)
