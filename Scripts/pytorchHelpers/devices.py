# Functions to assit with Pytorch pipelines involving device selection
# and assignment

# Load pytroch
import torch

# Detects and assigns a device for pytorch, preferncing GPUs
# Prints relevent system information
def assign():
    # Check Available Devices
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('Found {0} GPUs:'.format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            print('GPU {0}: {1}'.format(i, torch.cuda.get_device_name(i)))
    else:
        device = torch.device("cpu")
        print("Using " + str(cpuinfo.get_cpu_info()['brand']))
    return