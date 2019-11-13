"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

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
  accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1))
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  #batch_size = 32 # use the default settings?
  cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
  #n_classes = 10
  #dim_input = 3 * 32 * 32


  # Initiate test_x and test_y to use later when testing the network 
  test_x = cifar10['test'].images
  test_x = test_x.reshape(test_x.shape[0], -1)
  test_y = cifar10['test'].labels


  mlp = MLP(test_x.shape[1], dnn_hidden_units, test_y.shape[1], neg_slope)
  loss = CrossEntropyModule()

  train_accs, train_losses, test_accs, test_losses, epochs = [], [], [], [], []
  for epoch in range(FLAGS.max_steps):
      x, y = cifar10['train'].next_batch(FLAGS.batch_size)
      x = x.reshape(x.shape[0], -1)
      
      output = mlp.forward(x)
      dout = loss.backward(output, y)
      mlp.backward(dout)

      # Loop over all linear layers and update gradients.
      for lin_layer, _ in mlp.network:
          lin_layer.params['weight'] = lin_layer.params['weight'] - FLAGS.learning_rate * lin_layer.grads['weight']
          lin_layer.params['bias'] = lin_layer.params['bias'] - FLAGS.learning_rate * lin_layer.grads['bias']

      # Save losses for every 100th epoch
      if epoch % FLAGS.eval_freq == 0:
          epochs.append(epoch)
          train_out = mlp.forward(x)
          test_out = mlp.forward(test_x)

          train_acc = accuracy(train_out, y)
          test_acc = accuracy(test_out, test_y)

          train_loss = loss.forward(train_out, y)
          test_loss = loss.forward(test_out, test_y)

          train_accs.append(train_acc)
          test_accs.append(test_acc)
          train_losses.append(train_loss)
          test_losses.append(test_loss)
          print("=== Current epoch: {} ===".format(epoch))
          print("Train error: {} Validation error: {} Validation accuracy: {}".format(train_loss, test_loss, test_acc))

  # Plot accuracy and loss of every 100th epoch
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