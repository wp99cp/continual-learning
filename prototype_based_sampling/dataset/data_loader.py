from avalanche.benchmarks import (
    SplitCIFAR100,
    SplitMNIST,
    SplitFMNIST,
    SplitTinyImageNet,
)
from avalanche.logging import TensorboardLogger
from matplotlib import pyplot as plt
from torchvision import transforms


def save_example_data(
    dataset,
    tensorboard_logger: TensorboardLogger,
):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    for i, (x, y, t) in enumerate(dataset):
        if i >= 3:
            break

        ax = axs[i]
        x = (x + 2) / 4.0  # map to ~ [0, 1]
        x = x.clamp(0, 1)  # clamp to [0, 1]
        ax.imshow(x.permute(1, 2, 0), cmap="gray")
        ax.set_title(f"y={y}, t={t}")
        ax.axis("off")

    plt.tight_layout()
    tensorboard_logger.writer.add_figure("example_data", fig, global_step=0, close=True)


def print_stream_summary(stream):
    print(f"\n--- Stream: {stream.name}")
    for exp in stream:
        eid = exp.current_experience
        clss = exp.classes_in_this_experience
        tls = exp.task_labels
        print(f"  - EID={eid}, classes={clss}, tasks={tls}")
        print(f"     » contains {len(exp.dataset)} samples")


def load_dataset(
    dataset_name: str,
    print_summary: bool = True,
    n_experiences: int = 5,
    seed: int = 42,
    tensorboard_logger: TensorboardLogger = None,
):
    shared_base_args = {
        "n_experiences": n_experiences,  # 5 incremental experiences
        "seed": seed,  # fix the order of classes for reproducibility
        "return_task_id": True,  # add task labels
    }

    ###################################
    # load the dataset
    ###################################

    if dataset_name == "split_mnist":
        bm = SplitMNIST(**shared_base_args)
    elif dataset_name == "split_cifar100":
        bm = SplitCIFAR100(**shared_base_args)
    elif dataset_name == "split_fmnist":
        bm = SplitFMNIST(**shared_base_args)
    elif dataset_name == "split_tiny_imagenet":
        bm = SplitTinyImageNet(
            **shared_base_args,
            train_transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),  # to be consistent with CIFAR10/100
                    # default transformations for TinyImageNet
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
            eval_transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),  # to be consistent with CIFAR10/100
                    # default transformations for TinyImageNet
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        )

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    ###################################
    # print summary statistics
    ###################################

    if print_summary:
        print(f"\n###############\nDataset: {dataset_name}\n###############\n")
        print_stream_summary(bm.train_stream)
        print_stream_summary(bm.test_stream)

    ###################################
    # example data
    ###################################

    print(
        f"\n###############\nExample training data for the first experience:\n###############\n"
    )

    for i, (x, y, t) in enumerate(bm.train_stream[0].dataset):
        print(f"  - x.shape={x.shape}, y={y}, t={t}")
        if i > 2:
            break

    ###################################
    # disabled: we use the example images_samples_metrics from avalanche
    #
    # if tensorboard_logger is not None:
    #     save_example_data(bm.train_stream[0].dataset, tensorboard_logger)
    # else:
    #    print("No tensorboard logger provided, skipping example data visualization")

    return bm
