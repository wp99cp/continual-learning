import platform

from torch import cuda


def print_hardware_info(args):
    print(f"""

**********************************************************
* Geometry-Aware Sampling for Class-Incremental Learning *
**********************************************************

Starting experiment with the following configuration:
- Platform: {platform.platform()}
- Processor: {platform.processor()}
- CUDA Device: {cuda.get_device_name(args.cuda)}

**********************************************************


    """)
