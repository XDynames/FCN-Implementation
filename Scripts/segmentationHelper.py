# Like a dementor, but useful for segmentation tasks

# Built for translation of ground truth images into label arrays and vice verse in pytorch

import numpy as np

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