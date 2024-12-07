import platform

from torch import cuda


def print_hardware_info(args):
    print(
        f"""
####################
Hardware Information:
####################

- Platform: {platform.platform()}
- Processor: {platform.processor()}
- CUDA Device: {cuda.get_device_name(args.cuda)}

    """
    )
