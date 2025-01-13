from typing import Any, Type, Dict

from avalanche.core import Template
from avalanche.training.templates import SupervisedTemplate
from packaging.version import parse

import torch
from torch import Tensor
from torch.nn.functional import normalize
from torch.nn.modules import Module

from avalanche.training.utils import get_last_fc_layer, swap_last_fc_layer
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader

from geometric_aware_sampling.experiments.geometric_aware_sampling.geometric_balanced_buffer import (
    GeometricBalancedBuffer,
    GeometricBalancedBufferICarl,
)
from geometric_aware_sampling.experiments.geometric_aware_sampling.geometric_sampling_strategy import (
    BufferSamplingStrategy,
)
from geometric_aware_sampling.experiments.goldilocks.learning_speed_plugin import (
    LearningSpeedPlugin,
)


class CoPEGeometricPlugin(SupervisedPlugin, supports_distributed=False):
    """
    Experience replay plugin based in the Goldilocks buffer sampling strategy.
    See https://arxiv.org/abs/2406.09935 for more details.

    The `before_training_exp` callback is implemented in order to use the
    dataloader that creates mini-batches with examples from both training
    data and external memory. The examples in the mini-batch is balanced
    such that there are the same number of examples for each experience.

    The `after_training_exp` callback is implemented in order to add new
    patterns to the external memory based on the learning speed of the samples.
    Especially, we add samples that with a learning speed between the qth and
    s-th quantile of the learning speed distribution.

    In order to track the learning speed of the samples, we use the `LearningSpeedPlugin`
    plugin which adds a dimension to the data that contains the learning speed of the samples.
    The learning speed is updated on every mini-batch.

    :param replay_ratio: the ratio of replay samples to new samples in the mini-batch.
    :param mem_size: attribute controls the total number of samples to be stored
        in the external memory.
    :param task_balanced_dataloader: if True, buffer data loaders will be
        task-balanced, otherwise it will create a single dataloader for the
        buffer samples.
    :param upper_quantile: the upper quantile of the learning speed distribution that will
        never be included in the buffer
    :param lower_quantile: the lower quantile of the learning speed distribution that will
        never be included in the buffer
    :param q: ratio of training samples to keep, sampled using Goldilocks
    :param p: ratio of buffer samples to use 1.0 means that we can use all samples, as long as the replay_ratio
        is below q, this means that we replay a sample at most 1 time per epoch. If p is above 1.0, we treat
        p as a fixed value for the number of samples to replay per epoch.
    :param sample_per_epoch: if True, the plugin will sample new replay samples
        at the beginning of each epoch. Otherwise, it will sample new replay samples
        at the beginning of each experience.

    Based on the implementation of the ReplayPlugin from the Avalanche library.
    Release under the MIT License. Source:
    https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/plugins/replay.py

    """

    def __init__(
        self,
        sampling_strategy: Type[BufferSamplingStrategy],
        replay_ratio: float = 0.25,
        mem_size: int = 200,
        task_balanced_dataloader: bool = False,
        upper_quantile: float = 1 - 0.44,  # chosen according to the paper, figure 4 (b)
        lower_quantile: float = 0.12,  # chosen according to the paper, figure 4 (b)
        q: float = 0.4,
        p: float = 1.0,
        sample_per_epoch: bool = True,  # we want new samples per epoch
        T=0.1,
        alpha=0.99,
        storage_policy="LearningSpeed",
    ):
        super().__init__()
        self.initialized = False
        self.batch_size = None
        self.batch_size_mem = None
        self.mem_size = mem_size
        self.replay_ratio = replay_ratio
        self.task_balanced_dataloader = task_balanced_dataloader

        self.has_added_learning_speed_plugin = False

        if storage_policy == "LearningSpeed":
            # The storage policy samples the data based on the learning speed
            # and stores the samples in the external memory.
            self.storage_policy = GeometricBalancedBuffer(
                max_size=self.mem_size,
                adaptive_size=True,
                upper_quantile_ls=upper_quantile,
                lower_quantile_ls=lower_quantile,
                q=q,
                p=p,
                sampling_strategy=sampling_strategy,
            )
        elif storage_policy == "ICarl":
            # The storage policy samples the data based on the learning speed
            # and stores the samples in the external memory.
            self.storage_policy = GeometricBalancedBufferICarl(
                max_size=self.mem_size,
                adaptive_size=True,
                upper_quantile_ls=upper_quantile,
                lower_quantile_ls=lower_quantile,
                q=q,
                p=p,
                sampling_strategy=sampling_strategy,
            )
        else:
            raise ValueError("Unsupported storage policy " + storage_policy)

        # Operational memory: Prototypical memory
        # Scales with nb classes * feature size
        self.p_mem: Dict[int, Tensor] = {}
        self.tmp_p_mem = {}  # Intermediate to process batch for multiple times
        # PPP-loss
        self.T = T
        self.ppp_loss = PPPloss(self.p_mem, T=self.T)
        self.alpha = alpha

        self.classes = torch.empty(0)
        self.n_classes = 0
        self.class_id_to_idx = {}

        self.sample_per_epoch = sample_per_epoch
        self.task_idx = 0
        self.p_size = 100

    def before_training(self, strategy: Template, *args, **kwargs) -> Any:
        """Adds the learning speed plugin to the strategy."""

        """Enforce using the PPP-loss and add a NN-classifier."""
        if not self.initialized:
            strategy._criterion = self.ppp_loss
            print("Using the Pseudo-Prototypical-Proxy loss for CoPE.")

            # Normalize representation of last layer
            swap_last_fc_layer(
                strategy.model,
                torch.nn.Sequential(
                    get_last_fc_layer(strategy.model)[1], L2Normalization()
                ),
            )

            self._init_new_prototypes(
                torch.arange(0, self.n_classes).to(strategy.device)
            )

            self.initialized = True

        if self.has_added_learning_speed_plugin:
            return  # we need to add the LearningSpeedPlugin only once

        self.has_added_learning_speed_plugin = True

        if not any(isinstance(p, LearningSpeedPlugin) for p in strategy.plugins):
            strategy.plugins.append(LearningSpeedPlugin())
        else:
            print("WARNING: LearningSpeedPlugin already added to the strategy")

    @torch.no_grad()
    def _init_new_prototypes(self, targets: Tensor):
        """Initialize prototypes for previously unseen classes.
        :param targets: The targets Tensor to make prototypes for.
        """
        y_unique: Tensor = torch.unique(targets).squeeze().view(-1)
        for idx in range(y_unique.size(0)):
            c: int = y_unique[idx].item()
            if c not in self.p_mem:  # Init new prototype
                self.p_mem[c] = (
                    normalize(
                        torch.empty((1, self.p_size)).uniform_(-1, 1),
                        p=2,
                        dim=1,
                    )
                    .detach()
                    .to(targets.device)
                )
                self.class_id_to_idx[c] = len(self.class_id_to_idx)

    @torch.no_grad()
    def _update_running_prototypes(self, strategy):
        """Accumulate seen outputs of the network and keep counts."""
        y_unique = torch.unique(strategy.mb_y).squeeze().view(-1)
        for idx in range(y_unique.size(0)):
            c = y_unique[idx].item()
            idxs = torch.nonzero(strategy.mb_y == c).squeeze(1)
            p_tmp_batch = (
                strategy.mb_output[idxs].sum(dim=0).unsqueeze(0).to(strategy.device)
            )

            p_init, cnt_init = self.tmp_p_mem[c] if c in self.tmp_p_mem else (0, 0)
            self.tmp_p_mem[c] = (p_init + p_tmp_batch, cnt_init + len(idxs))

    def __sample_new_replay_samples(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs,
    ):

        # save buffer as local var to only call it once
        buffer = self.storage_policy.get_buffer(
            current_model=strategy.model, experience_dataset=strategy.experience.dataset
        )

        if len(buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

            # batch size split for task_idx > 1 (replay samples versus new samples)
        global_batch_size = strategy.train_mb_size

        assert (
            global_batch_size % (1.0 / self.replay_ratio) <= 1e-9
        ), "batch size must be divisible by (1 / replay_ratio)"

        batch_size = (
            int(global_batch_size * (1 - self.replay_ratio))
            if self.replay_ratio < 1
            else global_batch_size
        )
        batch_size_mem = int(global_batch_size * self.replay_ratio)

        assert strategy.adapted_dataset is not None

        other_dataloader_args = dict()

        if "ffcv_args" in kwargs:
            other_dataloader_args["ffcv_args"] = kwargs["ffcv_args"]

        if "persistent_workers" in kwargs:
            if parse(torch.__version__) >= parse("1.7.0"):
                other_dataloader_args["persistent_workers"] = kwargs[
                    "persistent_workers"
                ]

        dataset = ReplayDataLoader(
            strategy.adapted_dataset,
            buffer,
            oversample_small_tasks=False,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            **other_dataloader_args,
        )

        self.__print_statistics(strategy, buffer, dataset, global_batch_size)

        strategy.dataloader = dataset

    def __print_statistics(self, strategy, buffer, dataset, global_batch_size):
        print(
            f"{len(strategy.adapted_dataset)} new samples for task {self.task_idx} loaded."
        )

        replay_buffer_set = set()
        buffer_pool = self.storage_policy.replay_sampler.complete_buffer
        for x in buffer_pool:
            replay_buffer_set.add(x[3])
        print(f" » buffer pool contains {len(buffer_pool)} samples")
        print(
            f"   with {len(replay_buffer_set)} unique samples ({100 *(len(replay_buffer_set) / len(buffer_pool)):.2f}%)"
        )

        replay_set = set()
        # iterate over buffer
        for x in buffer:
            replay_set.add(x[3])  # unique sample id
        print(f" » {len(buffer)} samples selected for replay out of the buffer pool")
        print(
            f"   with {len(replay_set)} unique samples ({100 *(len(replay_set) / len(buffer)):.2f}%)"
        )

        full_set = set()

        for batch in dataset:
            for task, x in zip(batch[2], batch[3]):
                full_set.add(f"{x}__{task}")
        print(
            f" » {len(dataset) * global_batch_size} samples form the training set for task {self.task_idx}"
        )
        print(
            f"   with {len(full_set)} unique samples ({100 *(len(full_set) / (len(dataset) * global_batch_size)):.2f}%)"
        )

    def before_eval_forward(self, strategy: Template, *args, **kwargs) -> Any:
        """Has to create prototypes for new classes in the upcoming experience."""

        new_class = torch.unique(strategy.mb_y)
        self.classes = torch.unique(torch.cat([self.classes, new_class]))

        self._init_new_prototypes(self.classes)

    def before_training_epoch(self, strategy: "SupervisedTemplate", **kwargs):

        if self.sample_per_epoch:
            self.__sample_new_replay_samples(strategy, **kwargs)

    def before_training_exp(self, strategy: "SupervisedTemplate", **kwargs):

        if not self.sample_per_epoch:
            self.__sample_new_replay_samples(strategy, **kwargs)

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """
        uses after_training_exp to update the buffer after each training experience
        """

        self.task_idx += 1
        self.storage_policy.post_adapt(strategy, strategy.experience)
        if hasattr(self.storage_policy, "log_buffer_summary") and callable(
            getattr(self.storage_policy, "log_buffer_summary")
        ):
            self.storage_policy.log_buffer_summary(kwargs, self.task_idx)

        """After the current experience (batch), update prototypes and
        store observed samples for replay.
        """
        self._update_prototypes()  # Update prototypes
        self.storage_policy.update(strategy)  # Update memory

    @torch.no_grad()
    def _update_prototypes(self):
        """Update the prototypes based on the running averages."""
        for c, (p_sum, p_cnt) in self.tmp_p_mem.items():
            incr_p = normalize(p_sum / p_cnt, p=2, dim=1)  # L2 normalized
            old_p = self.p_mem[c].clone()
            new_p_momentum = (
                self.alpha * old_p + (1 - self.alpha) * incr_p
            )  # Momentum update
            self.p_mem[c] = normalize(new_p_momentum, p=2, dim=1).detach()
        self.tmp_p_mem = {}

    def after_forward(self, strategy, **kwargs):
        """
        After the forward we can use the representations to update our running
        avg of the prototypes. This is in case we do multiple iterations of
        processing on the same batch.

        New prototypes are initialized for previously unseen classes.
        """

        self._init_new_prototypes(strategy.mb_y)

        # Update batch info (when multiple iterations on same batch)
        self._update_running_prototypes(strategy)

    def after_eval_iteration(self, strategy, **kwargs):
        """Convert output scores to probabilities for other metrics like
        accuracy and forgetting. We only do it at this point because before
        this,we still need the embedding outputs to obtain the PPP-loss."""
        strategy.mb_output = self._get_nearest_neigbor_distr(strategy.mb_output)

    def _get_nearest_neigbor_distr(self, x: Tensor) -> Tensor:
        """
        Find closest prototype for output samples in batch x.
        :param x: Batch of network logits.
        :return: one-hot representation of the predicted class.
        """
        ns = x.size(0)
        nd = x.view(ns, -1).shape[-1]

        n_classes = len(self.p_mem)

        # Get prototypes
        seen_c = len(self.p_mem.keys())
        if seen_c == 0:  # no prototypes yet, output uniform distr. all classes
            return torch.Tensor(ns, n_classes).fill_(1.0 / n_classes).to(x.device)
        means = torch.ones(seen_c, nd).to(x.device) * float("inf")
        for c, c_proto in self.p_mem.items():
            means[self.class_id_to_idx[c]] = (
                c_proto  # Class idx gets allocated its prototype
            )

        # Predict nearest mean
        classpred = torch.LongTensor(ns)
        for s_idx in range(ns):  # Per sample
            dist = -torch.mm(means, x[s_idx].unsqueeze(-1))  # Dot product
            _, ii = dist.min(0)  # Min dist (no proto = inf)
            ii = ii.squeeze()
            classpred[s_idx] = ii.item()  # Allocate class idx

        # Convert to 1-hot
        out = torch.zeros(ns, n_classes).to(x.device)
        for s_idx in range(ns):
            out[s_idx, classpred[s_idx]] = 1
        return out  # return 1-of-C code, ns x nc


class L2Normalization(Module):
    """Module to L2-normalize the input. Typically used in last layer to
    normalize the embedding."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=1)


class PPPloss(object):
    """Pseudo-Prototypical Proxy loss (PPP-loss).
    This is a contrastive loss using prototypes and representations of the
    samples in the batch to optimize the embedding space.
    """

    def __init__(self, p_mem: Dict, T=0.1):
        """
        :param p_mem: dictionary with keys the prototype identifier and
                      values the prototype tensors.
        :param T: temperature of the softmax, serving as concentration
                  density parameter.
        """
        self.T = T
        self.p_mem = p_mem

    def __call__(self, x, y):
        """
        The loss is calculated with one-vs-rest batches Bc and Bk,
        split into the attractor and repellor loss terms.
        We iterate over the possible batches while accumulating the losses per
        class c vs other-classes k.
        """
        loss = None
        bs = x.size(0)
        x = x.view(bs, -1)  # Batch x feature size
        y_unique = torch.unique(y).squeeze().view(-1)
        include_repellor = len(y_unique.size()) <= 1  # When at least 2 classes

        # All prototypes
        p_y = torch.tensor([c for c in self.p_mem.keys()]).to(x.device).detach()
        if len(p_y) != 0:
            p_x = torch.cat([self.p_mem[c.item()] for c in p_y]).to(x.device).detach()

        for label_idx in range(y_unique.size(0)):  # Per-class operation
            c = y_unique[label_idx]

            # Make all-vs-rest batches per class (Bc=attractor, Bk=repellor set)
            Bc = x.index_select(0, torch.nonzero(y == c).squeeze(dim=1))
            Bk = x.index_select(0, torch.nonzero(y != c).squeeze(dim=1))

            p_idx = torch.nonzero(p_y == c).squeeze(dim=1)  # Prototypes
            pc = p_x[p_idx]  # Class proto
            pk = torch.cat([p_x[:p_idx], p_x[p_idx + 1 :]]).clone().detach()

            # Accumulate loss for instances of class c
            sum_logLc = self.attractor(pc, pk, Bc)
            sum_logLk = self.repellor(pc, pk, Bc, Bk) if include_repellor else 0
            Loss_c = -sum_logLc - sum_logLk  # attractor + repellor for class c
            loss = Loss_c if loss is None else loss + Loss_c  # Update loss
        return loss / bs  # Make independent batch size

    def attractor(self, pc, pk, Bc):
        """
        Get the attractor loss terms for all instances in xc.
        :param pc: Prototype of the same class c.
        :param pk: Prototoypes of the other classes.
        :param Bc: Batch of instances of the same class c.
        :return: Sum_{i, the part of same class c} log P(c|x_i^c)
        """
        m = torch.cat([Bc.clone(), pc, pk]).detach()  # Incl other-class proto
        pk_idx = m.shape[0] - pk.shape[0]  # from when starts p_k

        # Resulting distance columns are per-instance loss terms
        D = torch.mm(m, Bc.t()).div_(self.T).exp_()  # Distance matrix exp terms
        mask = torch.eye(*D.shape).bool().to(Bc.device)  # Exclude self-product
        Dm = D.masked_fill(mask, 0)  # Masked out products with self

        Lc_n, Lk_d = Dm[:pk_idx], Dm[pk_idx:].sum(dim=0)  # Num/denominator
        Pci = Lc_n / (Lc_n + Lk_d)  # Get probabilities per instance
        E_Pc = Pci.sum(0) / Bc.shape[0]  # Expectation over pseudo-prototypes
        return E_Pc.log_().sum()  # sum over all instances (sum i)

    def repellor(self, pc, pk, Bc, Bk):
        """
        Get the repellor loss terms for all pseudo-prototype instances in Bc.
        :param pc: Actual prototype of the same class c.
        :param pk: Prototoypes of the other classes (k).
        :param Bc: Batch of instances of the same class c. Acting as
        pseudo-prototypes.
        :param Bk: Batch of instances of other-than-c classes (k).
        :return: Sum_{i, part of same class c} Sum_{x_j^k} log 1 - P(c|x_j^k)
        """
        union_ck = torch.cat([Bc.clone(), pc, pk]).detach()
        pk_idx = union_ck.shape[0] - pk.shape[0]

        # Distance other-class-k to prototypes (pc/pk) and pseudo-prototype (xc)
        D = torch.mm(union_ck, Bk.t()).div_(self.T).exp_()

        Lk_d = D[pk_idx:].sum(dim=0).unsqueeze(0)  # Numerator/denominator terms
        Lc_n = D[:pk_idx]
        Pki = Lc_n / (Lc_n + Lk_d)  # probability

        E_Pk = (Pki[:-1] + Pki[-1].unsqueeze(0)) / 2  # Exp. pseudo/prototype
        inv_E_Pk = E_Pk.mul_(-1).add_(1).log_()  # log( (1 - Pk))
        return inv_E_Pk.sum()  # Sum over (pseudo-prototypes), and instances
