"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """
  class Flatten(nn.Module):
    def __init__(self):
        super(ConvNet.Flatten, self).__init__()

    def forward(self, x):
      #print("shape of x before flatten()", x.shape)
      x = x.view(x.size(0), -1)
      #print("shape of x after flatten()", x.shape)
      return x

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    #add relu + batchnorm2d
    super(ConvNet, self).__init__()
    kernel_size = 3
    #input channels = 3, output channels = 64
    maxpool1 = nn.MaxPool2d(kernel_size = kernel_size, stride = 2, padding = 1)
    batchnorm1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    conv1 = [nn.Conv2d(n_channels, 64, kernel_size = kernel_size, stride = 1, padding = 1), maxpool1, batchnorm1, nn.ReLU()]


    maxpool2 = nn.MaxPool2d(kernel_size = kernel_size, stride = 2, padding = 1)
    batchnorm2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    conv2 = [nn.Conv2d(64, 128, kernel_size = kernel_size, stride = 1, padding = 1), maxpool2, batchnorm2, nn.ReLU()]


    conv3a = nn.Conv2d(128, 256, kernel_size = kernel_size, stride = 1, padding = 1)
    conv3b = nn.Conv2d(256, 256, kernel_size = kernel_size, stride = 1, padding = 1)
    maxpool3 = nn.MaxPool2d(kernel_size = kernel_size, stride = 2, padding = 1)
    batchnorm3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    conv3 = [conv3a, conv3b, maxpool3,  batchnorm3, nn.ReLU()]

    conv4a = torch.nn.Conv2d(256, 512, kernel_size = kernel_size, stride = 1, padding = 1)
    conv4b = torch.nn.Conv2d(512, 512, kernel_size = kernel_size, stride = 1, padding = 1)
    maxpool4 = torch.nn.MaxPool2d(kernel_size = kernel_size, stride = 2, padding = 1)
    batchnorm4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    conv4 = [conv4a, conv4b, maxpool4, batchnorm4, nn.ReLU()]


    conv5a = torch.nn.Conv2d(512, 512, kernel_size = kernel_size, stride = 1, padding = 1)
    conv5b = torch.nn.Conv2d(512, 512, kernel_size = kernel_size, stride = 1, padding = 1)
    

    maxpool5 = torch.nn.MaxPool2d(kernel_size = kernel_size, stride = 2, padding = 1)
    batchnorm5 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    conv5 = [conv5a, conv5b, maxpool5, batchnorm5, nn.ReLU()]
    linear = torch.nn.Linear(512, n_classes)

    all_layers = [*conv1, *conv2, *conv3, *conv4, *conv5, ConvNet.Flatten(), linear]
    self.model = nn.Sequential(*all_layers)
    print(self.model)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = self.model(x)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out