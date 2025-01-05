from typing import Dict, List, Set, Union, TYPE_CHECKING, Iterable, Optional
from collections import defaultdict, OrderedDict

import torch
from torch import Tensor
from avalanche.evaluation import (
    Metric,
    _ExtendedGenericPluginMetric,
    _ExtendedPluginMetricValue,
)
from avalanche.evaluation.metric_utils import (
    default_metric_name_template,
    generic_get_metric_name,
)
from avalanche.evaluation.metrics import Mean

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


TrackedClassesType = Union[Dict[int, Iterable[int]], Iterable[int]]


class ClassForgetting(Metric[Dict[int, Dict[int, float]]]):
    """
    The Class Forgetting metric. This is a standalone metric
    used to compute more specific ones.

    Instances of this metric keep the running average forgetting
    over multiple <prediction, target> pairs of Tensors,
    provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average forgetting
    for all classes seen and across all predictions made since the last `reset`.
    The set of classes to be tracked can be reduced (please refer to the
    constructor parameters).

    The reset method will bring the metric to its initial state. By default,
    this metric in its initial state will return a
    `{task_id -> {class_id -> forgetting}}` dictionary in which all forgetting values are
    set to 0.
    """

    def __init__(self, classes: Optional[TrackedClassesType] = None):
        """
        Creates an instance of the standalone Forgetting metric.

        By default, this metric in its initial state will return an empty
        dictionary. The metric can be updated by using the `update` method
        while the running forgetting values can be retrieved using the `result` method.

        By using the `classes` parameter, one can restrict the list of classes
        to be tracked and in addition can immediately create plots for
        yet-to-be-seen classes.

        :param classes: The classes to keep track of. If None (default), all
            classes seen are tracked. Otherwise, it can be a dict of classes
            to be tracked (as "task-id" -> "list of class ids").
            By passing this parameter, the plot of each class is
            created immediately (with a default value of 0.0) and plots
            will be aligned across all classes. In addition, this can be used to
            restrict the classes for which the forgetting should be logged.
        """

        self.classes: Dict[int, int] = defaultdict(int)
        """
        A dictionary "class_id -> task_id"
        """

        self.dynamic_classes = False
        """
        If True, newly encountered classes will be tracked.
        """

        self._class_accuracies: Dict[int, Dict[int, Mean]] = defaultdict(
            lambda: defaultdict(Mean)
        )
        """
        A dictionary "class_id -> {task_id -> accuracy_on_this_class_id_after_training_on_this_task_id}".
        """

        if classes is not None:
            if isinstance(classes, dict):
                # Task-id -> classes dict
                self.classes = {
                    int(class_id): task_id
                    for task_id, class_list in classes.items()
                    for class_id in class_list
                }
        else:
            self.dynamic_classes = True

        self.__init_accs_for_known_classes()

    @torch.no_grad()
    def update(
        self,
        predicted_y: Tensor,
        true_y: Tensor,
        n_eval_stream: int,
    ) -> None:
        """
        Update the running accuracy (needed for forgetting)
        given the true and predicted labels for each class.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param n_eval_stream: the int number of evaluation streams associated to the current
            experience (assuming that it is increased in every experience)
        :return: None.
        """
        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        for pred, true in zip(predicted_y, true_y):

            if self.dynamic_classes:
                if int(true) not in self.classes.keys():
                    self.classes[int(true)] = n_eval_stream
            else:
                if int(true) not in self.classes.keys():
                    continue

            true_positives = (
                (pred == true).float().item()
            )  # 1.0 or 0.0 (correct or false prediction)
            # weight = 1 since it is one number (batch size of 16: updated 16 times)
            self._class_accuracies[int(true)][n_eval_stream].update(true_positives, 1)

    def result(self) -> Dict[int, Dict[int, float]]:
        """
        Retrieves the running forgetting value for each class.
        Forgetting = - BWT
        BWT (per class c) = accuracy on c after task k (latest task)
                            - accuracy on c after task j (first task containing c)

        Calling this method will not change the internal state of the metric.

        :return: A dictionary `{task_id -> {class_id -> running_forgetting}}`. The
            running forgetting value of each class is a float value between -1 and 1.
        """
        running_class_forgetting: Dict[int, Dict[int, float]] = dict()

        for class_id in sorted(self._class_accuracies.keys()):
            class_dict = self._class_accuracies[class_id]
            if len(class_dict.keys()) == 1:
                # can only compute forgetting starting from the second task
                continue
            for task_label in sorted(class_dict.keys()):
                # compute forgetting for any task after the task where a class first appears
                if task_label != self.classes[class_id]:
                    # add dictionary per task label if it does not exist yet
                    if not (task_label in running_class_forgetting.keys()):
                        running_class_forgetting[task_label] = dict()
                    running_class_forgetting[task_label][class_id] = (
                        class_dict[self.classes[class_id]].result()
                        - class_dict[task_label].result()
                    )

        return running_class_forgetting

    def reset(self) -> None:
        """
        Resets the metric. Not needed here since the accuracies for calculating
        forgetting are always needed, do not need to be reset (if you want to calculate
        forgetting, you always need the accuracies in the previous experiences)

        Still implemented since it is called by other functions.

        :return: None.
        """
        pass

    def reset_experience(self, experience_id):
        """
        Resets the accuracy values for the specified experience.

        :param experience_id: The experience for which the accuracy values should be reset

        :return: None.
        """
        for class_id in self._class_accuracies.keys():
            self._class_accuracies[class_id][experience_id].reset()

    def __init_accs_for_known_classes(self):
        for class_id, task_id in self.classes.items():
            self._class_accuracies[class_id][task_id].reset()

    def __str__(self):
        """
        For tracking purposes: print the accuracy values
        """
        class_accuracies_str = "; ".join(
            f"{class_id}, {experience_id}: {self._class_accuracies[class_id][experience_id].result()}"
            for class_id in self._class_accuracies.keys()
            for experience_id in self._class_accuracies[class_id].keys()
        )
        return class_accuracies_str


class ClassForgettingPluginMetric(_ExtendedGenericPluginMetric[ClassForgetting]):
    """
    Base class for all class forgetting plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, classes=None):
        super(ClassForgettingPluginMetric, self).__init__(
            ClassForgetting(classes=classes),
            reset_at=reset_at,
            emit_at=emit_at,
            mode=mode,
        )
        self.phase_name = "train"
        self.stream_name = "train"
        self.experience_id = 0

    def update(self, strategy: "SupervisedTemplate"):
        assert strategy.mb_output is not None
        assert strategy.experience is not None
        self._metric.update(
            strategy.mb_output, strategy.mb_y, len(strategy.current_eval_stream)
        )

        self.phase_name = "train" if strategy.is_training else "eval"
        self.stream_name = strategy.experience.origin_stream.name
        self.experience_id = strategy.experience.current_experience

    def before_eval(self, strategy):
        # important: accessing strategy.current_eval_stream only makes sense when
        # in evaluation mode (returns an empty list when training)
        self._metric.reset_experience(len(strategy.current_eval_stream))
        return super().before_eval(strategy)

    def after_eval_exp(self, strategy):
        return super().after_eval_exp(strategy)

    def result(self) -> List[_ExtendedPluginMetricValue]:
        metric_values = []
        task_forgetting = self._metric.result()

        for task_id, task_classes in task_forgetting.items():
            for class_id, class_forgetting in task_classes.items():
                metric_values.append(
                    _ExtendedPluginMetricValue(
                        metric_name=str(self),
                        metric_value=class_forgetting,
                        phase_name=self.phase_name,
                        stream_name=self.stream_name,
                        task_label=task_id,
                        experience_id=self.experience_id,
                        class_id=class_id,
                    )
                )

        return metric_values

    def metric_value_name(self, m_value: _ExtendedPluginMetricValue) -> str:
        m_value_values = vars(m_value)
        add_exp = self._emit_at == "experience"
        if not add_exp:
            del m_value_values["experience_id"]
        m_value_values["class_id"] = m_value.other_info["class_id"]

        return generic_get_metric_name(
            default_metric_name_template(m_value_values) + "/{class_id}",
            m_value_values,
        )


class ExperienceClassForgetting(ClassForgettingPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average forgetting over all patterns seen in that experience (separately
    for each class).

    This metric only works at eval time.
    """

    def __init__(self, classes=None):
        """
        Creates an instance of ExperienceClassForgetting metric
        """
        super().__init__(
            reset_at="experience",
            emit_at="experience",
            mode="eval",
            classes=classes,
        )

    def __str__(self):
        return "Top1_ClassForgetting_Exp"


class StreamClassForgetting(ClassForgettingPluginMetric):
    """
    At the end of the entire stream of experiences, this plugin metric
    reports the average forgetting over all patterns seen in all experiences
    (separately for each class).

    This metric only works at eval time.
    """

    def __init__(self, classes=None):
        """
        Creates an instance of StreamClassForgetting metric
        """
        super().__init__(
            reset_at="stream", emit_at="stream", mode="eval", classes=classes
        )

    def __str__(self):
        return "Top1_ClassForgetting_Stream"


def class_forgetting_metrics(
    *,
    experience=False,
    stream=False,
    classes=None,
) -> List[ClassForgettingPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param experience: If True, will return a metric able to log
        the per-class forgetting on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the per-class forgetting averaged over the entire evaluation stream of
        experiences.
    :param classes: The list of classes to track. See the corresponding
        parameter of :class:`ClassForgetting` for a precise explanation.

    :return: A list of plugin metrics.
    """
    metrics: List[ClassForgettingPluginMetric] = []

    if experience:
        metrics.append(ExperienceClassForgetting(classes=classes))

    if stream:
        metrics.append(StreamClassForgetting(classes=classes))

    return metrics


__all__ = [
    "TrackedClassesType",
    "ClassForgetting",
    "ExperienceClassForgetting",
    "StreamClassForgetting",
    "class_forgetting_metrics",
]
