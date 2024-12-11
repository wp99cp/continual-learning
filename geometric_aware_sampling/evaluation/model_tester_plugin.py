from avalanche.core import BaseSGDPlugin


class ModelTesterPlugin(BaseSGDPlugin):

    def __init__(self, callback, dataset):
        super().__init__()
        self.callback = callback
        self.dataset = dataset

    def after_training_exp(self, strategy, **kwargs):
        print("\nEvaluating model...")
        self.callback(strategy, self.dataset)
        print("Evaluation finished\n")

    def after_training_epoch(self, strategy, **kwargs):
        print("\nEvaluating model...")
        self.callback(strategy, self.dataset)
        print("Evaluation finished, resuming training...\n")
