# A Fully Convolutional Network built using VGG16 in pytorch, derviving the 
# FCN8, 16 and 32 architectures developed in Fully Convulotional Networks for
# Image Segmentation by J. Long etal.

# Load pytorch
import torch
import torchvision
import math
from torch import nn

# Build a FCN from pretrained VGG16
class VGG16FCN(nn.Module):
    # Initialises the FCN
    def __init__(self, numberOfClasses):
        super().__init__()
        
        # Load pretrained VGG16 to copy parameters from
        VGG16 = torchvision.models.vgg16(pretrained = True)
        
        # Downsampling Pathway
        self.downSample1 = nn.Sequential(*list(VGG16.features.children())[0:17])
        # Adjust first convulution layer to pad input by 100
        self.downSample1[0].padding = (100, 100)
        self.downSample2 = nn.Sequential(*list(VGG16.features.children())[17:24])
        # Demensionality Reduction to produce a tensor for each class
        self.downSample3 = nn.Sequential(*list(VGG16.features.children())[24:31])
        self.classification = nn.Sequential(
            # Oupt channel number is pulled from Shelhamer's github: 
            # master/voc-fcn8s/net.py
                nn.Conv2d(512, 4096, kernel_size = (1,1)), 
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Conv2d(4096, 4096, kernel_size = (1,1)),
                nn.ReLU(True),
                nn.Dropout(p=0.5),
                nn.Conv2d(4096, numberOfClasses, kernel_size = (1,1))
            )
        # Upsampling Pathway - Bilinear interpolation
        self.upSample2 = nn.Sequential(nn.Upsample(scale_factor = 2,
                                                            mode = 'bilinear'))
        self.upSample8 = nn.Sequential(nn.Upsample(scale_factor = 8,
                                                            mode = 'bilinear'))
        self.upSample16 = nn.Sequential(nn.Upsample(scale_factor = 16,
                                                            mode = 'bilinear'))
        self.upSample32 = nn.Sequential(nn.Upsample(scale_factor = 32,
                                                            mode = 'bilinear'))
        # Demensionality Reduction layers for each skip layer
        self.dimReduce1 = nn.Sequential(nn.Conv2d(512, numberOfClasses,
                                                      kernel_size = (1,1)))
        self.dimReduce2 = nn.Sequential(nn.Conv2d(256, numberOfClasses,
                                                      kernel_size = (1,1)))
        
    # Matches the demensions of two tesnors by padding
    def padUpToSize(self, tensorToPad, tensorB):
        diffX = tensorB.size()[2] - tensorToPad.size()[2]
        diffY = tensorB.size()[3] - tensorToPad.size()[3]
        tensorToPad = nn.functional.pad(tensorToPad, (diffY // 2,
                        math.ceil(diffY / 2), diffX // 2, math.ceil(diffX / 2)))
        return tensorToPad
        
# Derived Network versions using differen skip connections
class FCN8(VGG16FCN):                                     
    # Forward path through the network
    def forward(self, inputTensor):
        # Downsampling Pathway
        output1 = self.downSample1(inputTensor)
        output2 = self.downSample2(output1)
        output3 = self.downSample3(output2)
        output3 = self.classification(output3)
        
        # Upsampling Pathway
        # Skip for inclusion of pool4
        output3 = self.upSample2(output3)
        # Fix rounding of tensor size durring downsampling
        output3 = self.padUpToSize(output3, output2)
        # Reduce the tesnors channels to the number of classes
        output2 = self.dimReduce1(output2)
        # Sum the predictions from different depths
        output16 = torch.add(output3, output2)
        
        # Skip for inclusion of pool3
        output8 = self.upSample2(output16)
        # Fix rounding of tensor size durring downsampling
        output8 = self.padUpToSize(output8, output1)
        # Reduce the tesnors channels to the number of classes
        output1 = self.dimReduce2(output1)
        # Sum the predictions from different depths
        output8 = torch.add(output8, output1)
        # Upsample to return the orginal dimensions
        return self.upSample8(output8)
    
class FCN16(VGG16FCN):                                     
    # Forward path through the network
    def forward(self, inputTensor):
        # Downsampling Pathway
        output1 = self.downSample1(inputTensor)
        output1 = self.downSample2(output1)
        output2 = self.downSample3(output1)
        output2 = self.classification(output2)
        
        # Upsampling Pathway
        # Skip for inclusion of pool4
        output1 = self.upSample2(output1) 
        # Reduce the tesnors channels to the number of classes
        output1 = self.dimReduce1(output1)
        # Fix rounding of tensor size durring downsampling
        output1 = self.padUpToSize(output1, output2)
        # Sum the tensors
        output16 = torch.add(output1, output2)
        # Upsample to original size
        return self.upSample16(output16)
    
class FCN32(VGG16FCN):                                     
    # Forward path through the network
    def forward(self, inputTensor):
        # Downsampling Pathway
        output1 = self.downSample1(inputTensor)
        output1 = self.downSample2(output1)
        output1 = self.downSample3(output1)
        output1 = self.classification(output1)
        
        # Upsampling Pathway
        return self.upSample32(output1)
    

