from typing import Optional, Iterable, Any

import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor


def cm_image_creator(
    confusion_matrix_tensor: Tensor,
    display_labels: Optional[Iterable[Any]] = None,
    include_values=False,
    xticks_rotation=0,
    yticks_rotation=0,
    values_format=None,
    cmap="viridis",
    image_title="Confusion Matrix",
):
    """
    The default Confusion Matrix image creator.
    Code adapted from
    `Scikit learn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html>`_ # noqa

    :param confusion_matrix_tensor: The tensor describing the confusion matrix.
        This can be easily obtained through Scikit-learn `confusion_matrix`
        utility.
    :param display_labels: Target names used for plotting. By default, `labels`
        will be used if it is defined, otherwise the values will be inferred by
        the matrix tensor.
    :param include_values: Includes values in confusion matrix. Defaults to
        `False`.
    :param xticks_rotation: Rotation of xtick labels. Valid values are
        float point value. Defaults to 0.
    :param yticks_rotation: Rotation of ytick labels. Valid values are
        float point value. Defaults to 0.
    :param values_format: Format specification for values in confusion matrix.
        Defaults to `None`, which means that the format specification is
        'd' or '.2g', whichever is shorter.
    :param cmap: Must be a str or a Colormap recognized by matplotlib.
        Defaults to 'viridis'.
    :param image_title: The title of the image. Defaults to an empty string.
    :return: The Confusion Matrix as a PIL Image.

    Based on the original implementation in Avalanche:
    Source https://github.com/ContinualAI/avalanche/blob/625e46d9203878ed51457f6d7cd8d4e9fb05d093/avalanche/evaluation/metric_utils.py
    Released under the MIT License.

    """

    fig, ax = plt.subplots(dpi=600)

    cm = confusion_matrix_tensor.numpy()
    n_classes = cm.shape[0]
    im_ = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    if include_values:
        text_ = np.empty_like(cm, dtype=object)

        # print text with appropriate color depending on background
        thresh = (cm.max() + cm.min()) / 2.0

        for i in range(n_classes):
            for j in range(n_classes):
                color = cmap_max if cm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm[i, j], ".2g")
                    if cm.dtype.kind != "f":
                        text_d = format(cm[i, j], "d")
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm[i, j], values_format)

                text_[i, j] = ax.text(
                    j, i, text_cm, ha="center", va="center", color=color
                )

    fig.colorbar(im_, ax=ax)

    # we only want to show the numbers if there are less than 10 classes
    if display_labels is None and n_classes <= 10:
        display_labels = np.arange(n_classes)
    elif n_classes > 10:  # hide labels if there are more than 10 classes
        display_labels = None

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        ylabel="True label",
        xlabel="Predicted label",
        **(
            {"xticklabels": display_labels, "yticklabels": display_labels}
            if display_labels is not None
            else {}
        ),
    )

    if image_title != "":
        ax.set_title(image_title)

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
    plt.setp(ax.get_yticklabels(), rotation=yticks_rotation)

    fig.tight_layout()
    return fig
