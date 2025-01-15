from avalanche.training.plugins import CoPEPlugin
from avalanche.training.templates import SupervisedTemplate

from prototype_based_sampling.experiments.base_experiment import BaseExperimentStrategy


class PPPLossStrategy(BaseExperimentStrategy):
    """
    Strategy using Pseudo-Prototypical Proxy loss
    (https://openaccess.thecvf.com/content/ICCV2021/html/De_Lange_Continual_Prototype_Evolution_Learning_Online_From_Non-Stationary_Data_Streams_ICCV_2021_paper.html)
    Avalanche implementation:
    https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/plugins/cope.py
    Original implementation:
    https://github.com/Mattdl/ContinualPrototypeEvolution?tab=readme-ov-file

    Parameters for the plugin:
    mem_size = size of the buffer memory (prototypes + replay buffer)
    n_classes = number of classes
    p_size = dimension of the prototype
    alpha = prototypical momentum, set according to the paper (0.99 for MNIST and CIFAR10, 0.9 for CIFAR100)
    T = temperature parameter, set according to the paper (0.1 for MNIST and CIFAR10, 0.05 for CIFAR100)
    max_it_cnt = maximal number of iterations during the experience training
    """

    def create_cl_strategy(self):
        return SupervisedTemplate(
            **self.default_settings,
            plugins=[
                CoPEPlugin(
                    mem_size=100,
                    n_classes=10,
                    p_size=10,
                    alpha=0.99,
                    T=0.1,
                    max_it_cnt=1000,
                )
            ],
        )
