# Helper functions for creating dataloaders and data transforms
# for pytorch

import numpy as np
import math
import torch
import os


# Import some ojects from pytorch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

# Returns a list of pytorch dataloader objects with different properties for each phase of training
def buildDataloaders(batchSize, validationSetPercentage, pathToData, pathTrain, pathGT, pathTest,
                     numberOfWorkers = 0, transformerTrain = transforms.ToTensor(),
                     transformerTest = transforms.ToTensor()):
    
    # Get current working directory
    path =  os.getcwd() + pathToData
    # Make paths to the different image sets
    pathRaw = path + pathTrain
    pathGroundTruth = path + pathGT
    pathTesting = path + pathTest

    # Split images into training and validation randomly
    indicies = list(range(len(os.listdir(pathRaw))))
    splitSize = math.ceil(len(os.listdir(pathRaw))*validationSetPercentage) 
    valiIndices = np.random.choice(indicies, size = splitSize, replace = False)
    trainIndices = list(set(indicies) - set(valiIndices))

    # Create pytorch samplers using the generated indexs
    trainingSampler = torch.utils.data.SequentialSampler(trainIndices)
    valiSampler = torch.utils.data.SequentialSampler(valiIndices)
    # Using sequential sampler to hopefully get raw and groundtruth image pairs
    # from two different loaders


    # Define a list to store image loader objects with settings for different phases
    imageLoaders = {'train': DataLoader(ImageFolder(root = pathRaw, transform = transformerTrain),
                            batch_size = batchSize, num_workers = numberOfWorkers),

                    'trainGT': DataLoader(ImageFolder(root = pathGroundTruth, transform = transformerTest),
                                   batch_size = batchSize, num_workers = numberOfWorkers),

                    'vali': DataLoader(ImageFolder(root = pathRaw, transform = transformerTrain),
                            batch_size = batchSize, sampler = valiSampler, num_workers = numberOfWorkers),

                    'valiGT': DataLoader(ImageFolder(root = pathGroundTruth, transform = transformerTest),
                                  batch_size = batchSize, sampler = valiSampler, num_workers = numberOfWorkers),

                    'test': DataLoader(ImageFolder(root = pathTesting, transform = transformerTest),
                            batch_size = batchSize, num_workers = numberOfWorkers)
                   }
    
    return imageLoaders