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

    Instances of this metric keep the running accuracy (to compute forgetting)
    over multiple <prediction, target> pairs of Tensors,
    provided incrementally.
    The "prediction" and "target" tensors may contain plain labels or
    one-hot/logit vectors.

    Each time `result` is called, this metric emits the average forgetting
    for all classes seen and across all predictions made since the last `reset`.
    The set of classes to be tracked can be reduced (please refer to the
    constructor parameters).

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

        self._class_initial: Dict[int, Mean] = dict()
        """
        A dictionary "class_id -> initial accuracy measured".
        """

        self._class_last: Dict[int, Mean] = dict()
        """
        A dictionary "class_id -> most recent accuracy measured".
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

    @torch.no_grad()
    def update(self, predicted_y: Tensor, true_y: Tensor, exp_id: int) -> None:
        """
        Update the running accuracy (needed for forgetting)
        given the true and predicted labels for each class.

        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param exp_id: current experience id
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
                    self.classes[int(true)] = exp_id
            else:
                if int(true) not in self.classes.keys():
                    continue

            true_positives = (
                (pred == true).float().item()
            )  # 1.0 or 0.0 (correct or false prediction)

            if not (int(true) in self._class_initial.keys()):
                # set up initial
                self._class_initial[int(true)] = Mean()
                self._class_initial[int(true)].update(true_positives, 1)
            else:
                # in the first experience in which a class appears: update initial
                if exp_id == self.classes[int(true)]:
                    self._class_initial[int(true)].update(true_positives, 1)
                else:
                    # update last
                    if not (int(true) in self._class_last.keys()):
                        self._class_last[int(true)] = Mean()
                    # weight = 1 since it is one number (batch size of 16: updated 16 times)
                    self._class_last[int(true)].update(true_positives, 1)

    def result_class(self, c: int) -> Optional[float]:
        """
        Compute the forgetting for a specific class.

        :param c: the class for which to return forgetting

        :return: the difference between the first and last value encountered
            for c, if c is not None. It returns None if c has not been updated
            at least twice (i.e., value in the initial and last accuracy dict)
        """
        if c in self._class_initial and c in self._class_last:
            return self._class_initial[c].result() - self._class_last[c].result()
        else:
            return None

    def result(self) -> Dict[int, Dict[int, float]]:
        """
        Compute the forgetting for all classes.
        Forgetting = - BWT
        BWT (per class c) = accuracy on c after task k (latest task)
                            - accuracy on c after task j (initial task containing c)

        Calling this method will not change the internal state of the metric.

        :return: A dictionary `class_id -> running_forgetting`. The
            running forgetting value of each class is a float value between -1 and 1.
        """

        ik = set(self._class_initial.keys())
        both_keys = list(ik.intersection(set(self._class_last.keys())))

        forgetting: Dict[int, float] = {}
        for k in both_keys:
            forgetting[k] = (
                self._class_initial[k].result() - self._class_last[k].result()
            )

        return forgetting

    def reset_last(self, exp_id) -> None:
        """
        Resets the last value recorded for all classes.

        If reset is called inside an experience after the experience
            in which a class first appears, initial is never reset for this class

        :param exp_id: The current experience id

        :return: None
        """
        for class_id in self._class_initial.keys():
            if class_id in self._class_last.keys():
                self._class_last[class_id].reset()
            else:
                if exp_id != self.classes[class_id]:
                    continue
                self._class_initial[class_id].reset()

    def reset(self) -> None:
        """
        Resets the metric. Not needed here since reset_last can be used but it
        is still implemented since it is called by functions of the inherited classes.

        :return: None.
        """
        pass

    def __str__(self):
        """
        For tracking purposes: print the accuracy values

        One value displayed per class: only the initial value
        Two values displayed: initial and last value
        """
        class_accuracies_str = "; ".join(
            (
                f"{class_id}: {self._class_initial[class_id].result()}, {self._class_last[class_id].result()}"
                if class_id in self._class_last.keys()
                else f"{class_id}: {self._class_initial[class_id].result()}"
            )
            for class_id in self._class_initial.keys()
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

    def update(self, strategy: "SupervisedTemplate"):
        assert strategy.mb_output is not None
        assert strategy.experience is not None
        self._metric.update(
            strategy.mb_output, strategy.mb_y, strategy.clock.train_exp_counter
        )

        self.phase_name = "train" if strategy.is_training else "eval"
        self.stream_name = strategy.experience.origin_stream.name

    def before_eval(self, strategy):
        self._metric.reset_last(strategy.clock.train_exp_counter)
        return super().before_eval(strategy)

    def result(self) -> List[_ExtendedPluginMetricValue]:
        metric_values = []
        forgetting = self._metric.result()

        for class_id, class_forgetting in forgetting.items():
            metric_values.append(
                _ExtendedPluginMetricValue(
                    metric_name=str(self),
                    metric_value=class_forgetting,
                    phase_name=self.phase_name,
                    stream_name=self.stream_name,
                    experience_id=None,
                    task_label=None,
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
            reset_at="experience",  # no impact since reset() does nothing
            emit_at="experience",
            mode="eval",
            classes=classes,
        )

    def __str__(self):
        return "Top1_ClassForgetting_Exp"


def class_forgetting_metrics(
    *,
    experience=False,
    classes=None,
) -> List[ClassForgettingPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param experience: If True, will return a metric able to log
        the per-class forgetting on each evaluation experience.
    :param classes: The list of classes to track. See the corresponding
        parameter of :class:`ClassForgetting` for a precise explanation.

    :return: A list of plugin metrics.
    """
    metrics: List[ClassForgettingPluginMetric] = []

    if experience:
        metrics.append(ExperienceClassForgetting(classes=classes))

    return metrics


__all__ = [
    "TrackedClassesType",
    "ClassForgetting",
    "ExperienceClassForgetting",
    "class_forgetting_metrics",
]
