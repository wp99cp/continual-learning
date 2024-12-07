from avalanche.evaluation.metrics import (
    accuracy_metrics, loss_metrics, timing_metrics, forgetting_metrics, confusion_matrix_metrics
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def get_evaluator(n_classes):
    return EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(num_classes=n_classes, save_image=False, stream=True),
        loggers=[InteractiveLogger()],
        strict_checks=False
    )
