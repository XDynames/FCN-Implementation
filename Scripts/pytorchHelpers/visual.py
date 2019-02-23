# Functions to assist with visualising pytorch tensors

import numpy as np
from torchvision import utils

# Converts a pytorch tensor stack to a RGB colour image numpy array
def tensorToNumpy(imageTensor):
    grid = utils.make_grid(imageTensor)
    grid = grid.numpy().transpose((1, 2, 0))
    # clip any values less than 0 or greater than 1 to
    # avoid the wrath of matplotlib
    grid = np.clip(grid, 0, 1)
    return grid