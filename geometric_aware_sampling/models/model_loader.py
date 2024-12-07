from geometric_aware_sampling.models.SlimResNet18 import SlimResNet18


def load_model(
    model_name: str = "slim_resnet18",
    cl_dataset=None,
    compile_model: bool = True,
):
    if cl_dataset is None:
        raise ValueError("cl_dataset must be provided to load_model")

    if model_name == "slim_resnet18":
        model = SlimResNet18(
            input_dim=cl_dataset.train_stream[0].dataset[0][0].shape,
            nclasses=cl_dataset.n_classes,
        )
    else:
        raise ValueError(f"Model {model_name} not supported")

    # compile the model to speed up training
    # see https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    if compile_model:
        model.compile()
    else:
        print("Model not compiled. Training might be slower.")

    return model
