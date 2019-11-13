"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
from mlp_pytorch import torch
import cifar10_utils
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '800,400,200,100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 512
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None
dtype = torch.FloatTensor
device =  torch.device('cpu')

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  accuracy = (predictions.argmax(dim=1) == targets.argmax(dim=1)).type(dtype).mean()
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy.item()

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)
  torch.manual_seed(42)
  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope
  lr = FLAGS.learning_rate
  ########################
  # PUT YOUR CODE HERE  #
  #######################
  cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)

  # load x_test and y_test (test data)
  x_test = cifar10["test"].images
  y_test = cifar10["test"].labels

  x_test = x_test.reshape(x_test.shape[0], -1)

  # convert np arrays into torch tensors 

  x_test = torch.tensor(x_test, requires_grad = False).type(dtype).to(device) #10000 x 3072 (32x32x3)
  y_test = torch.tensor(y_test, requires_grad = False).type(dtype).to(device) #10000 x 10

  MLP_model = MLP(n_inputs = 32*32*3, n_hidden = dnn_hidden_units, n_classes = 10, neg_slope = neg_slope)
  loss = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(MLP_model.parameters(), lr = 0.00005, weight_decay=1e-6)
  
  max_steps = FLAGS.max_steps
  train_accs, train_losses, test_accs, test_losses, epochs = [], [], [], [], []
  for epoch in range(max_steps):
      MLP_model.train()
      x, y = cifar10['train'].next_batch(FLAGS.batch_size)
      x = x.reshape(x.shape[0], -1)
      x = torch.from_numpy(x).type(dtype).to(device)
      y = torch.from_numpy(y).type(dtype).to(device)

      optimizer.zero_grad()
      out = MLP_model.forward(x)

      loss_training = loss(out, y.argmax(dim=1))
      
      #backward propagation
      loss_training.backward()
      optimizer.step()

      
      if epoch % FLAGS.eval_freq == 0:
        epochs.append(epoch)
        train_loss = loss_training
        train_losses.append(train_loss)
        train_accs.append(accuracy(out, y))

        # now test on x_test, y_test
        out = MLP_model.forward(x_test)
        test_acc = accuracy(out, y_test)
        test_accs.append(test_acc)
        test_loss = loss(out, y_test.argmax(dim=1))
        test_losses.append(test_loss)

        print(" === Current epoch {} ===".format(epoch))
        print("Train error: {} Validation error: {} Validation accuracy: {}".format(train_loss, test_loss, test_acc))

  plt.plot(epochs, train_accs, label='train_acc' )
  plt.plot(epochs, test_accs, label='test_acc')
  plt.title("Train vs test accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend(loc = 'upper left')
  plt.show()

  plt.clf()

  plt.plot(epochs, train_losses, label='train_loss' )
  plt.plot(epochs, test_losses, label='test_loss' )
  plt.title("Train vs test loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(loc = 'upper right')
  plt.show()

  # To write:
  # Test different parameters, save results and plot them later
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  FLAGS, unparsed = parser.parse_known_args()

  main()