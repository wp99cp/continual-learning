import argparse

import torch
import torch.optim.lr_scheduler
from avalanche.training.supervised import FromScratchTraining
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from geometric_aware_sampling.dataset.data_loader import load_dataset
from geometric_aware_sampling.evaluation.evaluation import get_evaluator
from geometric_aware_sampling.models.model_loader import load_model


def run(args: argparse.Namespace):
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    # set the default precision for matrix multiplication to high
    torch.set_float32_matmul_precision('high')

    ###################################
    # Load the dataset and model
    ###################################

    dataset_name = "split_mnist"  # (split_mnist or split_cifar100)

    cl_dataset = load_dataset(
        dataset_name,
        print_summary=True,  # print summary statistics of the dataset / experience
        save_example_input=True  # save example data to a file
    )

    model = load_model(
        model_name="slim_resnet18",  # (currently only slim_resnet18)
        cl_dataset=cl_dataset,
    )

    optimizer = Adam(
        model.parameters(),
        betas=(0.9, 0.999),
        lr=0.001,
        weight_decay=1e-5
    )

    criterion = CrossEntropyLoss()

    ###################################
    # Continual Learning Strategy
    ###################################

    cl_strategy = FromScratchTraining(

        # model and optimizer (normal PyTorch modules)
        model=model,
        optimizer=optimizer,
        criterion=criterion,

        # number of training epochs per experience
        train_epochs=1,

        # batch sizes
        train_mb_size=16,
        eval_mb_size=16,

        device=device,
        evaluator=get_evaluator(cl_dataset.n_classes),
    )

    ###################################
    # Start the experiment
    ###################################

    print("\n\n####################\nStarting experiment\n####################\n\n")

    results = []
    for i, experience in enumerate(cl_dataset.train_stream, 1):
        print(f"\n - Experience {i} of {cl_dataset.n_experiences}")
        print("    Current classes:", experience.classes_in_this_experience)
        print(f"    » {cl_strategy.__class__.__name__} strategy\n")

        cl_strategy.train(experience)

        print('Computing accuracy on the whole test set')
        results.append(cl_strategy.eval(cl_dataset.test_stream))

    print("\n\n####################\nExperiment finished\n####################\n\n")

    ###################################
    # print confusion matrix as image
    ###################################

    # get the confusion matrix from the evaluator
    confusion_matrix = cl_strategy.evaluator.get_all_metrics()['ConfusionMatrix_Stream/eval_phase/test_stream']
    confusion_matrix = confusion_matrix[1]

    for i, cm in enumerate(confusion_matrix):
        cm = cm.numpy()
        cm = cm / cm.sum(axis=1, keepdims=True)

        # plot the confusion matrix
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')

        plt.savefig(f'confusion_matrix_{i}.png')

        # clear the plot
        plt.clf()
