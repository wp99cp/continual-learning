from typing import Sequence, Tuple, List

from avalanche.evaluation.metrics import (
    accuracy_metrics,
    confusion_matrix_metrics,
    forgetting_metrics,
    loss_metrics,
    timing_metrics,
    class_accuracy_metrics,
    images_samples_metrics,
    ImagesSamplePlugin,
    bwt_metrics,
)
from avalanche.logging import BaseLogger
from avalanche.training.plugins import EvaluationPlugin
from torch import Tensor

from geometric_aware_sampling.evaluation.cm_image_creator import cm_image_creator


def _load_data(
    self, strategy: "SupervisedTemplate"
) -> Tuple[List[Tensor], List[int], List[int]]:
    assert strategy.adapted_dataset is not None
    dataloader = self._make_dataloader(strategy.adapted_dataset, strategy.eval_mb_size)

    images: List[Tensor] = []
    labels: List[Tensor] = []
    tasks: List[Tensor] = []

    for batch_images, batch_labels, batch_tasks, _ in dataloader:
        n_missing_images = self.n_wanted_images - len(images)
        labels.extend(batch_labels[:n_missing_images].tolist())
        tasks.extend(batch_tasks[:n_missing_images].tolist())
        images.extend(batch_images[:n_missing_images])
        if len(images) == self.n_wanted_images:
            return images, labels, tasks
    return images, labels, tasks


# PATCH avalanche library
ImagesSamplePlugin._load_data = _load_data


def get_evaluator(n_classes, loggers: BaseLogger | Sequence[BaseLogger] = None):
    return EvaluationPlugin(
        ##########################
        # model performance metrics
        ##########################
        accuracy_metrics(epoch=True, experience=True, stream=True),
        class_accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        ##########################
        # Continual Learning metrics
        ##########################
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        confusion_matrix_metrics(
            num_classes=n_classes,
            save_image=True,
            normalize="pred",
            stream=True,
            image_creator=cm_image_creator,
            absolute_class_order=False,  # sort by class introduction
        ),
        ##########################
        # print example input data to tensorboard
        ##########################
        images_samples_metrics(),
        ##########################
        # system metrics
        ##########################
        # the following metrics make the training process very slow
        # disk_usage_metrics(epoch=True, experience=True, stream=True),
        # cpu_usage_metrics(epoch=True, experience=True, stream=True),
        # gpu_usage_metrics(epoch=True, experience=True, stream=True, gpu_id=0),
        # ram_usage_metrics(epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, experience=True, stream=True),
        ##########################
        # logger
        ##########################
        loggers=loggers,
        strict_checks=False,
    )
