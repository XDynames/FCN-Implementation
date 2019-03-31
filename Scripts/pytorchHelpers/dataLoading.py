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
def buildDataloaders(batchSize, validationSetPercentage, pathToData, pathTrain,
                     pathGT, pathTest, transformerTrain,
                     transformerGT, transformerTest, numberOfWorkers = 0):
    
    # Get current working directory
    path =  os.getcwd() + pathToData
    # Make paths to the different image sets
    pathRaw = path + pathTrain
    pathGroundTruth = path + pathGT
    pathTesting = path + pathTest
    
    imageSets = {'train': ImageFolder(root = pathRaw, transform = transformerTrain),
                 'trainGT': ImageFolder(root = pathGroundTruth, transform = transformerGT),
                 'trainVali': ImageFolder(root = pathRaw, transform = transformerTrain),
                 'trainValiGT':ImageFolder(root = pathGroundTruth, transform = transformerGT),
                 'vali': ImageFolder(root = pathRaw, transform = transformerTrain),
                 'valiGT': ImageFolder(root = pathGroundTruth, transform = transformerGT),
                 'test':  ImageFolder(root = pathTesting, transform = transformerTest) }

    # Split images into training and validation randomly
    indicies = list(range(len(imageSets['train'])))
    splitSize = math.ceil(len(indicies)*validationSetPercentage) 
    valiIndices = np.random.choice(indicies, size = splitSize, replace = False)
    trainIndices = list(set(indicies) - set(valiIndices))
    # Create pytorch samplers using the generated indexs
    trainingSampler = torch.utils.data.SequentialSampler(trainIndices)
    valiSampler = torch.utils.data.SequentialSampler(valiIndices)
    # Using sequential sampler to get raw and groundtruth image pairs
    # from two different loaders

    # Define a list to store image loader objects with settings for different phases
    imageLoaders = {'train': DataLoader(imageSets['train'], batch_size = batchSize,
                            sampler = trainingSampler, num_workers = numberOfWorkers),

                    'trainGT': DataLoader(imageSets['trainGT'], batch_size = batchSize,
                              sampler = trainingSampler, num_workers = numberOfWorkers),
                    
                    'trainVali': DataLoader(imageSets['trainVali'], batch_size = batchSize,
                            sampler = trainingSampler, num_workers = numberOfWorkers),

                    'trainValiGT': DataLoader(imageSets['trainValiGT'], batch_size = batchSize,
                              sampler = trainingSampler, num_workers = numberOfWorkers),

                    'vali': DataLoader(imageSets['vali'], batch_size = batchSize,
                            sampler = valiSampler, num_workers = numberOfWorkers),

                    'valiGT': DataLoader(imageSets['valiGT'], batch_size = batchSize,
                            sampler = valiSampler, num_workers = numberOfWorkers),

                    'test': DataLoader(imageSets['test'], batch_size = batchSize,
                            num_workers = numberOfWorkers)
                   }
    
    return imageLoaders