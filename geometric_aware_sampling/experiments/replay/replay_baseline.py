from avalanche.training import Naive, Replay

from geometric_aware_sampling.experiments.base_experiment import BaseExperimentStrategy


class ReplayBaseline(BaseExperimentStrategy):
    """

    Baseline experiment using the Naive continual learning strategy
    from the Avalanche library.

    """

    def create_cl_strategy(self):
        return Replay(
            # model and optimizer (normal PyTorch modules)
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            # number of training epochs per experience
            train_epochs=5,
            # batch sizes
            train_mb_size=16,
            eval_mb_size=16,
            device=self.device,
            evaluator=self.eval_plugin,
        )
