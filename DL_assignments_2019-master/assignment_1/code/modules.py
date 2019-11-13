"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    """
    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}
    
    # initiate weights and biases

    self.params["weight"] = np.random.normal(loc = 0.0, scale = 0.00001, size = (out_features, in_features))
    self.params["bias"] = np.zeros((out_features, 1))

    self.grads["weight"] = np.zeros((out_features, in_features))
    self.grads["bias"] = np.zeros((out_features, 1))"""
    self.params = {'weight': np.random.normal(0, 0.0001, (out_features, in_features)), 'bias': np.zeros((out_features, 1))}
    self.grads = {'weight': np.zeros((out_features, in_features)), 'bias': np.zeros((out_features))}
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.old_x = x
    out = self.params["weight"] @ x.T + self.params["bias"]
    ########################
    # END OF YOUR CODE    #
    #######################

    return out.T

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.grads['weight'] = dout.T @ self.old_x 
    self.grads['bias'] = np.mean(dout, axis=0).reshape(self.params['bias'].shape)
    dx = dout @ self.params['weight']
    ########################
    # END OF YOUR CODE    #
    #######################
    return dx

class LeakyReLUModule(object):
  """
  Leaky ReLU activation module.
  """
  def __init__(self, neg_slope):
    """
    Initializes the parameters of the module.

    Args:
      neg_slope: negative slope parameter.

    TODO:
    Initialize the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.neg_slope = neg_slope
    ########################
    # END OF YOUR CODE    #
    #######################
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.old_x = x
    self.out = np.maximum(0, x) + self.neg_slope * np.minimum(0, x)
    ########################
    # END OF YOUR CODE    #
    #######################

    return self.out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    deriv_x = np.ones_like(self.old_x)
    deriv_x[self.old_x < 0] = self.neg_slope
    dx = np.multiply(dout, deriv_x)
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx


class SoftMaxModule(object):
  """
  Softmax activation module.
  """

  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    max_x = x.max(axis = 1)
    max_x = max_x.reshape((max_x.shape[0], 1))
    numerator = np.exp(x - max_x)

    denominator = numerator.sum(axis=1)
    denominator = denominator.reshape((denominator.shape[0], 1))

    self.out = numerator/denominator
    ########################
    # END OF YOUR CODE    #
    #######################
    return self.out

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = np.zeros(dout.shape)
    for i, element in enumerate(self.out):
      temp_deriv = np.diagflat(element) - np.outer(element, element)
      dx[i] = dout[i] @ temp_deriv
    #######################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """

  def forward(self, x, y):
    """
    Forward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = np.sum(y * (-1) * np.log(x), axis=1).mean()
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = -np.divide(y,x)/len(y)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx