from typing import Any

import torch
from avalanche.benchmarks import AvalancheDataset
from avalanche.training import ExemplarsBuffer
from torch import Tensor


class WeightedSamplingBuffer(ExemplarsBuffer):
    """

    Buffer updated with reservoir sampling.

    TODO: check the behavior of the buffer when the buffer is full and new data
        is added. e.g. after the third epoch, the buffer is full and new data
        is added. What happens to the buffer? How is data discarded?

        Check if this is correctly implemented according to the paper...

    """

    def __init__(self, max_size: int):
        """
        :param max_size:
        """

        super().__init__(max_size)

        # INVARIANT: _buffer_weights is always sorted.
        self._buffer_weights = torch.zeros(0)

    def post_adapt(self, agent, exp):
        """Update buffer."""
        self.update_from_dataset(exp.dataset)

    def log_buffer_summary(self, kwargs, task_idx: int, buffer_name: str = "buffer"):
        tensorboard_logger = kwargs["tensorboard_logger"].writer
        tensorboard_logger.add_scalar(buffer_name + "size", len(self.buffer), task_idx)
        tensorboard_logger.close()

    def update_from_dataset(self, new_data: AvalancheDataset, weights: Tensor = None):
        """Update the buffer using the given dataset.

        :param weights:
        :param new_data:
        :return:
        """

        if weights is None:
            new_weights = torch.rand(len(new_data))
        else:
            assert len(weights) == len(new_data)
            new_weights = weights

        cat_weights = torch.cat([new_weights, self._buffer_weights])
        cat_data = new_data.concat(self.buffer)
        sorted_weights, sorted_idxs = cat_weights.sort(descending=True)

        buffer_idxs = sorted_idxs[: self.max_size]
        self.buffer = cat_data.subset(buffer_idxs)
        self._buffer_weights = sorted_weights[: self.max_size]

    def resize(self, strategy: Any, new_size: int):
        """Update the maximum size of the buffer."""
        self.max_size = new_size

        if len(self.buffer) <= self.max_size:
            return

        self.buffer = self.buffer.subset(torch.arange(self.max_size))
        self._buffer_weights = self._buffer_weights[: self.max_size]
