# Calulates the 1-d size of the conv layer's output given an input tensor
def dimSizeConv(inputSize, kernelSize, padding, stride):
    dim = (inputSize - kernelSize + 2 * padding) / stride + 1
    return dim

# Function that calulates the output tensor from a convulutionlayer
# given an inputtensor, kernel size, stride and padding
def convolutionOutputSize(inputSize, kernelSize, padding = [0, 0], stride = [1, 1]):
    width = dimSizeConv(inputSize[0], kernelSize[0], padding[0], stride[0])
    height = dimSizeConv(inputSize[1], kernelSize[1], padding[1], stride[1])
    return [width, height]

# Calulates the 1-d size of the maxpool layer's output given an input tensor
def dimSizeMPool(inputSize, kernelSize, dilation, padding, stride):
    dim = (inputSize + 2 * padding - kernelSize - (kernelSize - 1) * 
               (dilation - 1)) / stride + 1
    return dim

# Fucntion that calulates the output tensor from a maxpooling layer
# given the input tensor, kenrel size, stride and padding
def maxpoolOutputSize(inputSize, kernelSize, dilation, padding = [0, 0]):
    stride = kernelSize
    width = dimSizeMPool(inputSize[0], kernelSize[0], dilation, padding[0], stride[0])
    height = dimSizeMPool(inputSize[1], kernelSize[1], dilation, padding[1], stride[1])
    return [width, height]