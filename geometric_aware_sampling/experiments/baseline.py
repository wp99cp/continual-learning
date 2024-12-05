import argparse

import torch
import torch.optim.lr_scheduler
from avalanche.benchmarks import SplitCIFAR100
from avalanche.models import SlimResNet18
from avalanche.training.supervised import FromScratchTraining
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from geometric_aware_sampling.utils.evaluation import get_evaluator


def run(args: argparse.Namespace):
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    benchmark = SplitCIFAR100(
        n_experiences=5,
        seed=42,
        return_task_id=False,

    )

    model = SlimResNet18(nclasses=100)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    # compile the model to speed up training
    model.compile()

    cl_strategy = FromScratchTraining(

        # model and optimizer (normal PyTorch modules)
        model=model,
        optimizer=optimizer,
        criterion=criterion,

        # number of training epochs per experience
        train_epochs=2,

        # batch sizes
        train_mb_size=32,
        eval_mb_size=32,

        device=device,
        evaluator=get_evaluator(),
    )

    print("\n\nStart Experiment Run\n")

    # TRAINING LOOP
    for i, experience in enumerate(benchmark.train_stream, 1):
        print(f"Experience {i} of {benchmark.n_experiences}")
        print("Current classes:", experience.classes_in_this_experience)
        print(f" » {cl_strategy.__class__.__name__} strategy")

        print("\n")
        cl_strategy.train(experience)
        cl_strategy.eval(benchmark.test_stream[:i])

        print("\nEnd of experience %d\n\n" % i)
