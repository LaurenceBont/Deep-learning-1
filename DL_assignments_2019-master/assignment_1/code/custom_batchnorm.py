import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object. 
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormAutograd, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.betas = nn.Parameter(torch.zeros(n_neurons))
    self.gammas = nn.Parameter(torch.ones(n_neurons))
    self.eps = eps
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Implement batch normalization forward pass as given in the assignment.
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #print("shape of input = {}".format(input.shape)) # 128x4
    assert len(input.shape) == 2, "Shape is wrong" # should we do more checks
    assert self.betas.shape == self.gammas.shape, "Shape of gamma or beta is wrong"
    #dim = 0 want mean over column, per batch, per neuron dus dan shrink je je 128x4 naar 1x4, dus de mean per batch per neuron
    mean = input.mean(dim = 0)
    var = input.var(dim = 0, unbiased = False)
    #print("Current mean: {} \n Current var: {}".format(mean, var))
    norm = (input - mean) / torch.sqrt(var + self.eps)
    out = self.gammas * norm + self.betas
    ########################
    # END OF YOUR CODE    #
    #######################
    return out



######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
  pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This makes sure that the context objects
  are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
      beta: mean bias tensor, applied per neuron, shpae (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor

    TODO:
      Implement the forward pass of batch normalization
      Store constant non-tensor objects via ctx.constant=myconstant
      Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
      Intermediate results can be decided to be either recomputed in the backward pass or to be stored
      for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    #print("shape of input = {}".format(input.shape)) # 128x4
    assert len(input.shape) == 2, "Shape is wrong" # should we do more checks
    assert beta.shape == gamma.shape, "Shape of gamma or beta is wrong"
    mean = input.mean(dim = 0)
    var = input.var(dim = 0, unbiased = False)
    #print("Current mean: {} \n Current var: {}".format(mean, var))
    norm = (input - mean) / torch.sqrt(var + eps)
    out = gamma * norm + beta

    ctx.save_for_backward(input, beta, gamma, mean, var) # we need to save out, betas, gammas, mean, var (and eps)
    ctx.eps = eps
    ########################
    # END OF YOUR CODE    #
    #######################

    return out


  @staticmethod
  def backward(ctx, grad_output):
    """
    Compute backward pass of the batch normalization.
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments
    
    TODO:
      Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
      Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
      inputs to None. This should be decided dynamically.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    input, beta, gamma, mean, var = ctx.saved_tensors
    eps = ctx.eps
    input_needs_grad, gamma_needs_grad, beta_needs_grad = ctx.needs_input_grad

    x_hat = (input - mean) * (1/torch.sqrt(var + eps))
    dx_hat = grad_output * gamma
    
    grad_beta = grad_output.sum(0) if beta_needs_grad else None
    grad_gamma = (grad_output * x_hat).sum(0) if gamma_needs_grad else None 
    grad_input = 1./input.shape[0] * (1/torch.sqrt(var + eps)) * (input.shape[0]*dx_hat - dx_hat.sum(0) - x_hat*(dx_hat*x_hat).sum(0)) if input_needs_grad else None
    ########################
    # END OF YOUR CODE    #
    #######################

    # return gradients of the three tensor inputs and None for the constant eps
    return grad_input, grad_gamma, grad_beta, None



######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward is called.
  The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormManualModule, self).__init__()
    self.betas = nn.Parameter(torch.zeros(n_neurons))
    self.gammas = nn.Parameter(torch.ones(n_neurons))
    self.eps = eps
    ########################
    # PUT YOUR CODE HERE  #
    #######################

    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Instantiate a CustomBatchNormManualFunction.
      Call it via its .apply() method.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    assert len(input.shape) == 2, "Shape is wrong"
    assert self.betas.shape == self.gammas.shape, "Shape of gamma or beta is wrong"
    batchNormFunc = CustomBatchNormManualFunction()
    out = batchNormFunc.apply(input, self.gammas, self.betas, self.eps)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out






if __name__=='__main__':
    # create test batch
    n_batch = 128
    n_neurons = 4
    # create random tensor with variance 2 and mean 3
    x = 2*torch.randn(n_batch, n_neurons, requires_grad=True)+10
    print('Input data:\n\tmeans={}\n\tvars={}'.format(x.mean(dim=0).data, x.var(dim=0).data))

    # test CustomBatchNormAutograd
    print('3.1) Test automatic differentation version')
    bn_auto = CustomBatchNormAutograd(n_neurons)
    y_auto = bn_auto(x)
    print('\tmeans={}\n\tvars={}'.format(y_auto.mean(dim=0).data, y_auto.var(dim=0).data))
    # test CustomBatchNormManualFunction
    # this is recommended to be done in double precision
    print('3.2 b) Test functional version')
    input = x.double()
    gamma = torch.sqrt(10*torch.arange(n_neurons, dtype=torch.float64, requires_grad=True))
    beta = 100*torch.arange(n_neurons, dtype=torch.float64, requires_grad=True)
    bn_manual_fct = CustomBatchNormManualFunction(n_neurons)
    y_manual_fct = bn_manual_fct.apply(input, gamma, beta)
    print('\tmeans={}\n\tvars={}'.format(y_manual_fct.mean(dim=0).data, y_manual_fct.var(dim=0).data))
    # gradient check
    grad_correct = torch.autograd.gradcheck(bn_manual_fct.apply, (input,gamma,beta))
    if grad_correct:
        print('\tgradient check successful')
    else:
        raise ValueError('gradient check failed')

    # test CustomBatchNormManualModule
    print('3.2 c) Test module of functional version')
    bn_manual_mod = CustomBatchNormManualModule(n_neurons)
    y_manual_mod = bn_manual_mod(x)
    print('\tmeans={}\n\tvars={}'.format(y_manual_mod.mean(dim=0).data, y_manual_mod.var(dim=0).data))