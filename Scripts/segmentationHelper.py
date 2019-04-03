# Like a dementor, but useful for segmentation tasks

# Built for translation of ground truth images into label arrays and vice verse in pytorch

import numpy as np
import torch

class Segmentor:
    
    # Initialise the segmentor with a corresponding list of colours
    # labels and names ie. colours[0] -> lables[0] -> names[0]
    def __init__(self, colours, labels, names):
        self._colours = [[0, 0, 0]] + colours
        self._labels = [0] + labels
        self._names = ['Background'] + names
    

    # Maps the list of colours in the image to their respective
    # labels and retuns the resulting label array
    def mapColoursToLabels(self, image): 
        labeled = np.zeros((image.shape[0], image.shape[1]), dtype = np.uint8)
        # Check if the pixel matches any of the labeled colours      
        for colour, label in zip(self._colours, self._labels):
            mask = np.all(image == colour, axis = -1)
            labeled = np.where(mask == True, label, labeled)
        return labeled

    # Wrapper for mapColoursToLabels so it can be directly used as a pytorch
    # transform, taking in a PIL image and return a numpy array of labels.
    # torchvision.transforms.Lambda(lambda x : Segmentor.mapColoursToLabelsPIL(x))
    def mapColoursToLabelsPIL(self, imagePIL): 
        image = np.array(imagePIL)
        labeled = self.mapColoursToLabels(image)
        return labeled
    
    
    # From an numpy array of labels creates an RGB numpy array image
    # using the original colours from the ground truth
    def mapLabelsToColours(self, labelArray):
        # Constrcuts a colour image using the labels in the array
        return np.array(self._colours)[labelArray]
    
    
    # mapLabelsToColours that can be used on pytorch tensors
    # directly
    def mapLabelsToColoursPytensor(self, tensor):
        # Extract label array as a numpy array and
        # move it to the CPU
        labelArray = tensor.cpu().numpy()
        return self.mapLabelsToColours(labelArray)
    
    # Counts the number of correctly labeled pixels in a prediction
    def countCorrectPixelLabels(self, batchOfPredictions, batchOfGT):        
        # Return how many times pixels where correctly labeled as a numpy array
        return torch.sum(batchOfPredictions == batchOfGT.data).cpu().numpy()
    
    def pixelwiseAccuracy(self, batchOfOutput, batchOfGT):
        # extract predictions by taking the maximum across the class channels
        # and return tensors specifying the channel the max is in
        batchOfPredictions = torch.argmax(batchOfOutput, dim = 1)
        # Calculate the number of correctly labeled pixels
        correctPixels = self.countCorrectPixelLabels(batchOfPredictions, batchOfGT)
        # Divide by the number of predictions made
        return correctPixels / np.prod(batchOfPredictions.shape)
    
    # Calculates the Intersection over Union of a prediction with GT
    def iou(self, prediction, GT):
        # Initialise a list to store each classes IoU
        IoUs = []
        # Don't consider background label
        for classLabel in self._labels[1:]:
            # Create masks with 1s where the label is found
            predictionPositions = prediction == classLabel
            GTPositions = GT == classLabel
            # Calculate the intersection and the union
            intersection = np.logical_and(predictionPositions, GTPositions)
            union = np.logical_or(predictionPositions, GTPositions)
            # Append the result
            IoUs.append(np.sum(intersection)/np.sum(union))
        # Return the average IoU across all classes
        return np.mean(np.array(IoUs))
    
    # Calculates the average intersection over union for a batch
    def intersectionOverUnion(self, batchOfOutput, batchOfGT):
        # extract predictions by taking the maximum across the class channels
        # and return tensors specifying the channel the max is in
        batchOfPredictions = torch.argmax(batchOfOutput, dim = 1)
        # Initialise a list to store each images IoU
        IoUs = []
        # Calculate the IoU for each prediction
        for prediction, GT in zip(batchOfPredictions, batchOfGT):
            IoUs.append(self.iou(prediction.cpu().numpy(), GT.cpu().numpy()))
        # Return the averaged IoU for the batch
        return np.mean(np.array(IoUs))