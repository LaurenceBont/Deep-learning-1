"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
      neg_slope: negative slope parameter for LeakyReLU

    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(MLP, self).__init__()
    #print(n_inputs, n_hidden, n_classes, neg_slope)
    #self.layers = nn.Sequential(nn.Linear(n_inputs, int(n_hidden)), nn.modules.LeakyReLU(neg_slope), nn.Softmax())
    #print(self.layers)

    self.network = []
    for h in n_hidden:
      
      hidden_layer = nn.Linear(n_inputs, h)
      batch_layer = nn.BatchNorm1d(h)
      hidden_activation = nn.LeakyReLU(neg_slope)
      
      self.network.append(hidden_layer)
      self.network.append(batch_layer)
      self.network.append(hidden_activation)
      n_inputs = h 

    output_layer = nn.Linear(n_inputs,n_classes)
    output_activiation = nn.Softmax()
    self.network.append(output_layer)
    #self.network.append(nn.Dropout(0.2))
    self.network.append(output_activiation)
    self.model = nn.Sequential(*self.network)
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
