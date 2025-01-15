from collections import defaultdict
from typing import Dict, List, Union, TYPE_CHECKING

import torch
from avalanche.evaluation import (
    Metric,
    _ExtendedGenericPluginMetric,
    _ExtendedPluginMetricValue,
)
from avalanche.evaluation.metric_utils import (
    default_metric_name_template,
    generic_get_metric_name,
)
from torch import Tensor

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class ClassificationTracker(Metric[Dict[str, bool]]):

    def __init__(self):

        self.target_hits = {}

    @torch.no_grad()
    def update(
        self,
        inputs: Tensor,
        predicted_y: Tensor,
        true_y: Tensor,
        task_labels: Union[int, Tensor],
    ) -> None:
        """

        Stores 0 or 1 in the self.target_hits[key] where key = [task_id]__[class_id]__[sample_id]
        - 1 if the prediction is correct
        - 0 if the prediction is wrong

        :param inputs:  The input data.
        :param predicted_y: The model prediction. Both labels and logit vectors
            are supported.
        :param true_y: The ground truth. Both labels and one-hot vectors
            are supported.
        :param task_labels: the int task label associated to the current
            experience or the task labels vector showing the task label
            for each pattern.
        :return: None.
        """

        if isinstance(task_labels, int):
            task_labels = torch.full_like(true_y, task_labels)

        for input, task_label, pred, true in zip(
            inputs, task_labels, predicted_y, true_y
        ):

            task_label = int(task_label.item())
            # TODO: use sample indexes instead of hash of the input
            sample_id = str(hash(input.cpu().numpy().tobytes()))
            pred = int(torch.argmax(pred).item())
            true = int(true.item())

            key = f"{task_label}__{true}__{sample_id}"
            self.target_hits[key] = 1 if pred == true else 0

    def result(self) -> Dict[str, bool]:
        """
        Returns the raw data collected by the metric.
        """
        return self.target_hits

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """

        self.target_hits = defaultdict(dict)


class ClassificationTrackerPluginMetric(
    _ExtendedGenericPluginMetric[ClassificationTracker]
):
    """
    Base class for all class accuracy plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode):
        super(ClassificationTrackerPluginMetric, self).__init__(
            ClassificationTracker(),
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
            strategy.mb_x, strategy.mb_output, strategy.mb_y, strategy.mb_task_id
        )

        self.phase_name = "train" if strategy.is_training else "eval"
        self.stream_name = strategy.experience.origin_stream.name
        self.experience_id = strategy.experience.current_experience

    def result(self) -> List[_ExtendedPluginMetricValue]:
        metric_values = []
        task_accuracies = self._metric.result()

        # iterate over key, value pairs
        for key, class_accuracy in task_accuracies.items():

            task_id, class_id, sample_id = key.split("__")

            metric_values.append(
                _ExtendedPluginMetricValue(
                    metric_name=str(self),
                    metric_value=class_accuracy,
                    phase_name=self.phase_name,
                    stream_name=self.stream_name,
                    task_label=int(task_id),
                    experience_id=self.experience_id,
                    class_id=int(class_id),
                    sample_id=int(sample_id),
                )
            )

        return metric_values

    def metric_value_name(self, m_value: _ExtendedPluginMetricValue) -> str:
        m_value_values = vars(m_value)
        add_exp = self._emit_at == "experience"
        if not add_exp:
            del m_value_values["experience_id"]
        m_value_values["class_id"] = m_value.other_info["class_id"]
        m_value_values["sample_id"] = m_value.other_info["sample_id"]

        return generic_get_metric_name(
            default_metric_name_template(m_value_values) + "/{class_id}/{sample_id}",
            m_value_values,
        )


class StreamClassificationTracker(ClassificationTrackerPluginMetric):

    def __init__(self):
        super().__init__(reset_at="stream", emit_at="stream", mode="eval")

    def __str__(self):
        return "ClassificationTracker_Stream"


class ExperienceClassificationTracker(ClassificationTrackerPluginMetric):

    def __init__(self):
        super().__init__(
            reset_at="experience",
            emit_at="experience",
            mode="eval",
        )

    def __str__(self):
        return "ClassificationTracker_Exp"


def classification_tracker(
    *,
    experience=False,
    stream=False,
) -> List[ClassificationTrackerPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.


    :param experience: If True, will return a metric able to log
        the per-class accuracy on each evaluation experience.
    :param stream: If True, will return a metric able to log
        the per-class accuracy averaged over the entire evaluation stream of
        experiences.

    :return: A list of plugin metrics.
    """
    metrics: List[ClassificationTrackerPluginMetric] = []

    if experience:
        metrics.append(ExperienceClassificationTracker())

    if stream:
        metrics.append(StreamClassificationTracker())

    return metrics
