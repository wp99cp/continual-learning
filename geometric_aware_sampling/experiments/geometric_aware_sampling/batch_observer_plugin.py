from typing import Any

import numpy as np
from avalanche.core import SupervisedPlugin, Template
from matplotlib import pyplot as plt


class BatchObserverPlugin(SupervisedPlugin, supports_distributed=False):
    """ "
    A plugin that observes the batch size and batch composition of the data
    """

    def __init__(self, normalize_steps: bool = True, strategy_name: str = None):
        super().__init__()

        self.normalize = normalize_steps
        self.strategy_name = strategy_name

        self.full_batch_composition_history = {}
        self.batch_composition = {}

        self.full_different_composition_history = {}
        self.different_composition = {}

        self.history_length = 0

        self.batch_sizes = np.array([])

        # a simple lookup table mapping class labels to task ids
        self.task_idx = 0
        self.task_lookup = {}
        self.iteration_counter = 0

    def after_training_iteration(self, strategy: Template, *args, **kwargs):
        self.batch_sizes = np.append(self.batch_sizes, len(strategy.mb_y))

        for i, _label in enumerate(strategy.mb_y):
            label = _label.item()
            if label not in self.batch_composition:
                self.batch_composition[label] = 0
                self.different_composition[label] = set()
            self.batch_composition[label] += 1
            self.different_composition[label].add(strategy.mbatch[3][i].item())

        self.iteration_counter += 1

    def before_training_epoch(self, strategy: Template, *args, **kwargs) -> Any:
        self.batch_composition = {}
        self.different_composition = {}
        self.iteration_counter = 0

    def after_training_epoch(self, strategy: Template, *args, **kwargs):

        if self.normalize:
            normalization_factor = self.iteration_counter
        else:
            normalization_factor = 1.0

        for label, count in self.batch_composition.items():
            if label not in self.full_batch_composition_history:
                # initialize the history for this label back-filling with zeros
                self.full_batch_composition_history[label] = [0] * self.history_length
                self.full_different_composition_history[label] = [
                    0
                ] * self.history_length

                # add to task lookup on first encounter
                self.task_lookup[label] = self.task_idx

            self.full_batch_composition_history[label].append(
                float(count) / normalization_factor
            )
            self.full_different_composition_history[label].append(
                float(len(self.different_composition[label])) / normalization_factor
            )

        # back-fill with zeros for labels that were not encountered in this epoch
        for label in self.full_batch_composition_history.keys():
            if label not in self.batch_composition:
                self.full_batch_composition_history[label].append(0)
                self.full_different_composition_history[label].append(0)

        self.history_length += 1

    def after_training_exp(self, strategy: Template, *args, **kwargs) -> Any:
        print(
            f"Batch Size {np.mean(self.batch_sizes):.4f}, std: {np.std(self.batch_sizes):.4f}, "
            f"min: {np.min(self.batch_sizes)}, max: {np.max(self.batch_sizes)}"
        )
        self.batch_sizes = np.array([])

    def __print(
        self,
        history,
        kwargs,
        legend_y_label,
        title,
    ):
        # Create a stacked plot of the batch composition
        fig, ax = plt.subplots(figsize=(10, 5), dpi=250)

        # Initialize an array to hold the cumulative sum
        cumulative_counts = np.zeros_like(next(iter(history.values())))

        # Define a base colormap for each task
        assert len(set(self.task_lookup.values())) <= 5
        task_colormaps = {
            0: plt.cm.Blues,
            1: plt.cm.Greens,
            2: plt.cm.Oranges,
            3: plt.cm.Purples,
            4: plt.cm.Reds,
        }

        # Keep track of how many classes each task has to assign shades
        task_class_counts = {task_id: 0 for task_id in set(self.task_lookup.values())}
        for task_id in self.task_lookup.values():
            task_class_counts[task_id] += 1

        # Track which tasks have been added to the legend
        plotted_tasks = set()

        for label, counts in history.items().__reversed__():
            task_id = self.task_lookup[label]
            class_idx = (
                list(self.task_lookup.keys()).index(label) % task_class_counts[task_id]
            )

            # Use a specific colormap and generate shades
            colormap = task_colormaps[task_id]
            color = colormap((class_idx + 1) / (task_class_counts[task_id] + 1))

            # Plot the current class
            ax.fill_between(
                range(len(counts)),
                cumulative_counts,
                cumulative_counts + counts,
                color=color,
                step="post",
                label=f"Task {task_id}" if task_id not in plotted_tasks else None,
            )
            cumulative_counts += counts
            plotted_tasks.add(task_id)

        ax.set_xlabel("epoch idx")
        ax.set_ylabel(legend_y_label)
        ax.legend(loc="upper center", ncol=5)

        if title:
            ax.set_title(title)

        # add figure to tensorboard
        tensorboard_logger = kwargs["tensorboard_logger"].writer
        tensorboard_logger.add_figure(title, fig, global_step=self.task_idx, close=True)

    def after_training(self, strategy: Template, *args, **kwargs) -> Any:
        self.task_idx += 1

        strategy_name = strategy.__class__.__name__

        if self.strategy_name is not None:
            strategy_name = self.strategy_name

        self.__print(
            self.full_batch_composition_history,
            kwargs,
            "samples per class/task",
            (
                f"Class/Task Composition of a Minibatch ({strategy_name})"
                if self.normalize
                else f"Class/Task Composition of a Epoch ({strategy_name})"
            ),
        )
        self.__print(
            self.full_different_composition_history,
            kwargs,
            "unique samples per class/task",
            (
                f"Unique Samples per Class/Task in a Minibatch ({strategy_name})"
                if self.normalize
                else f"Unique Samples per Class/Task in an Epoch ({strategy_name})"
            ),
        )
