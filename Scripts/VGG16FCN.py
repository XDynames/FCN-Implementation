# A Fully Convolutional Network built using VGG16 in pytorch, derviving the 
# FCN8, 16 and 32 architectures developed in Fully Convulotional Networks for
# Image Segmentation by J. Long etal.

# Load pytorch
import torch
import torchvision

# Build a FCN from pretrained VGG16
class VGG16FCN(torch.nn.Module):
    # Initialises the FCN
    def __init__(self, numberOfClasses):
        super().__init__()
        
        # Load pretrained VGG16 to copy parameters from
        VGG16 = torchvision.models.vgg16(pretrained = True)
        
        # Downsampling Pathway
        self.downSample1 = torch.nn.Sequential(*list(VGG16.features.children())[0:17]) # Dim / 8
        self.downSample2 = torch.nn.Sequential(*list(VGG16.features.children())[17:24]) # Dim / 16
        # Demensionality Reduction to produce a tensor for each class
        self.downSample3 = torch.nn.Sequential(
                *list(VGG16.features.children())[24:31],
                torch.nn.Conv2d(256, numberOfClasses, kernel_size = (1,1)), 
                torch.nn.ReLU(True),
                torch.nn.Dropout(p=0.5),
                torch.nn.Conv2d(256, numberOfClasses, kernel_size = (1,1)),
                torch.nn.ReLU(True),
                torch.nn.Dropout(p=0.5),
                torch.nn.Conv2d(256, numberOfClasses, kernel_size = (1,1))
        ) # Dim / 32
        
        # Upsampling Pathway - Bilinear interpolation
        self.upSample2 = torch.nn.Sequential(torch.nn.Upsample(scale_factor = 2, mode = 'bilinear'))
        self.upSample8 = torch.nn.Sequential(torch.nn.Upsample(scale_factor = 8, mode = 'bilinear'))
        self.upSample16 = torch.nn.Sequential(torch.nn.Upsample(scale_factor = 16, mode = 'bilinear'))
        self.upSample32 = torch.nn.Sequential(torch.nn.Upsample(scale_factor = 32, mode = 'bilinear'))

# Derived Network versions using differen skip connections
class FCN8(VGG16FCN):                                     
    # Forward path through the network
    def forward(self, inputTensor):
        # Downsampling Pathway
        output1 = self.downSample1(inputTensor)
        output2 = self.downSample2(output1)
        output3 = self.downSample3(output2)
        # Upsampling Pathway
        output16 = self.upSample2(output3) + output2
        output8 = self.upSample2(output16) + output1
        return self.upSample8(output8)
    
class FCN16(VGG16FCN):                                     
    # Forward path through the network
    def forward(self, inputTensor):
        # Downsampling Pathway
        output1 = self.downSample1(inputTensor)
        output1 = self.downSample2(output1)
        output2 = self.downSample3(output1)
        # Upsampling Pathway
        output = self.upSample2(output2) + output1
        return self.upSample16(output)
    
class FCN32(VGG16FCN):                                     
    # Forward path through the network
    def forward(self, inputTensor):
        # Downsampling Pathway
        output1 = self.downSample1(inputTensor)
        output1 = self.downSample2(output1)
        output1 = self.downSample3(output1)
        # Upsampling Pathway
        return self.upSample32(output1)