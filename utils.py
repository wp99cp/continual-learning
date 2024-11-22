from datetime import datetime

from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    confusion_matrix_metrics,
    images_samples_metrics,
    labels_repartition_metrics,
class_accuracy_metrics
)
from avalanche.logging import TensorboardLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def get_evaluator():
    return EvaluationPlugin(
        accuracy_metrics(
            stream=True,
            experience=True,
        ),
        class_accuracy_metrics(
            stream=True,
            experience=True,
        ),
        images_samples_metrics(
            on_train=True,
            on_eval=True,
            n_cols=10,
            n_rows=5,
        ),
        labels_repartition_metrics(
            # image_creator=repartition_bar_chart_image_creator,
            on_train=True,
            on_eval=True,
        ),
        accuracy_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        loss_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(
            stream=True, class_names=[str(i) for i in range(10)]
        ),

        loggers=[
            TensorboardLogger(f"tb_data/{datetime.now()}"),
            InteractiveLogger(),
        ],
    )
