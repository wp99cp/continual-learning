from typing import Sequence

from avalanche.evaluation.metrics import (
    accuracy_metrics,
    confusion_matrix_metrics,
    forgetting_metrics,
    loss_metrics,
    timing_metrics,
)
from avalanche.logging import BaseLogger
from avalanche.training.plugins import EvaluationPlugin


def get_evaluator(n_classes, loggers: BaseLogger | Sequence[BaseLogger] = None):
    return EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        confusion_matrix_metrics(
            num_classes=n_classes,
            save_image=True,
            normalize="all",
            stream=True,
            absolute_class_order=False,  # sort by class introduction
        ),
        forgetting_metrics(experience=True, stream=True),
        timing_metrics(epoch=True),
        loggers=loggers,
        strict_checks=False,
    )
