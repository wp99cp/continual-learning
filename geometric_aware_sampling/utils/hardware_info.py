import platform

from torch import cuda


def print_hardware_info(args):
    device = cuda.get_device_name(args.cuda) if cuda.is_available() else "cpu"
    print(
        f"""
####################
Hardware Information:
####################

- Platform: {platform.platform()}
- Processor: {platform.processor()}
- CUDA Device: {device}

    """
    )
