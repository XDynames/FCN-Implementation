# Functions to assist with visualising pytorch tensors

import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils

# Converts a pytorch tensor stack to a RGB colour image numpy array
def tensorToNumpy(imageTensor):
    if len(imageTensor.size()) == 4:
        grid = utils.make_grid(imageTensor)
        grid = grid.numpy()
        # Rearrange Channels
        grid = grid.transpose((1, 2, 0))
    else:
        grid = imageTensor.numpy()
        flattened = grid[0,:,:]
        for image in grid[1:]:
            flattened = np.hstack((flattened, image))
        grid = flattened
    return grid

# Displays an image from an inputted tensor
def plotOutputTensor(tensor):
    plt.figure(figsize=(16, 14))
    plt.axis('off')
    for number1, pred in enumerate(tensor):
        pred = pred.detach().to("cpu")
        for number2, channel in enumerate(pred):
            channel = tensorToNumpy(channel)
            plt.imshow(channel)
            plt.savefig('Figures\\Prediction-'+str(number1)+
                        '-channel-'+str(number2)+'.png')
    return

# Displays an image from an inputted tensor
def plotPredictionTensor(tensor):
    plt.figure(figsize=(16, 14))
    plt.axis('off')
    for number1, pred in enumerate(tensor):
        pred = pred.detach().to("cpu")
        pred = tensorToNumpy(pred)
        plt.imshow(pred)
        plt.savefig('Figures\\PredictionOverall-' + str(number1) + '.png')
    return