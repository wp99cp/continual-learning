from datetime import datetime

from avalanche.evaluation.metrics import (
    accuracy_metrics, loss_metrics, timing_metrics, forgetting_metrics, confusion_matrix_metrics
)
from avalanche.logging import TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin


def get_evaluator(n_classes):
    return EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        confusion_matrix_metrics(
            num_classes=n_classes,
            save_image=True,
            normalize="all",
            stream=True,
            absolute_class_order=False  # sort by class introduction
        ),
        forgetting_metrics(experience=True, stream=True),

        timing_metrics(epoch=True),
        loggers=[
            TensorboardLogger(f"tb_data/{datetime.now()}"),
        ],
        strict_checks=False
    )
