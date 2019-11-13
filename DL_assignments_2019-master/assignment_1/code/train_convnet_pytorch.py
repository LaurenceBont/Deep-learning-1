"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
from mlp_pytorch import torch
import pickle
# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000 #5000
EVAL_FREQ_DEFAULT = 500 #500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
dtype = torch.FloatTensor
device =  torch.device('cpu')

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
  accuracy = (predictions.argmax(dim=1) == targets.argmax(dim=1)).type(dtype).mean()
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)

  # load x_test and y_test (test data)
  x_test = cifar10["test"].images
  y_test = cifar10["test"].labels

  #x_test = x_test.reshape(x_test.shape[0], -1)

  # convert np arrays into torch tensors 

  x_test = torch.tensor(x_test, requires_grad = False).type(dtype).to(device) #10000 x 3072 (32x32x3)
  y_test = torch.tensor(y_test, requires_grad = False).type(dtype).to(device) #10000 x 10

  model = ConvNet(3,10)
  loss = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr = FLAGS.learning_rate, weight_decay=1e-6)
  
  max_steps = FLAGS.max_steps
  train_accs, train_losses, test_accs, test_losses, steps = [], [], [], [], []
  for step in range(max_steps):
      #print(epoch)
      model.train()
      x, y = cifar10['train'].next_batch(FLAGS.batch_size)
      #x = x.reshape(x.shape[0], -1)
      x = torch.tensor(x).type(dtype).to(device)
      y = torch.tensor(y).type(dtype).to(device)

      out = model.forward(x)
      
      loss_training = loss(out, y.argmax(dim=1))
      
      #backward propagation
      loss_training.backward()
      optimizer.step()

      
      if step % FLAGS.eval_freq == 0:
        steps.append(step)
        train_loss = loss_training
        train_losses.append(train_loss)
        train_accs.append(accuracy(out, y))

        # now test on x_test, y_test
        out = model.forward(x_test)
        test_acc = accuracy(out, y_test)
        test_accs.append(test_acc)
        test_loss = loss(out, y_test.argmax(dim=1))
        test_losses.append(test_loss)

        print(" === Current step {} ===".format(step))
        print("Train error: {} Validation error: {} Validation accuracy: {}".format(train_loss, test_loss, test_acc))

  with open('outfile', 'wb') as fp:
    pickle.dump([train_accs, train_losses, test_accs, test_losses, steps], fp)
 
  plt.plot(epochs, train_accs, label='train_acc' )
  plt.plot(epochs, test_accs, label='test_acc')
  plt.title("Train vs test accuracy")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy")
  plt.legend(loc = 'upper left')
  plt.savefig('cnn_train_test_acc.png')

  plt.show()

  plt.clf()

  plt.plot(epochs, train_losses, label='train_loss' )
  plt.plot(epochs, test_losses, label='test_loss' )
  plt.title("Train vs test loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(loc = 'upper right')
  plt.savefig('cnn_train_test_loss.png')
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
  FLAGS, unparsed = parser.parse_known_args()

  main()