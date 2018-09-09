# Multilayer Neural Network for Exclusive OR (XOR) Function

import numpy as np
import random

w1 = np.loadtxt('corrected_weights1.txt')
w2 = np.loadtxt('corrected_weights2.txt', ndmin=2)

def sigmoid(s):
  # activation function
  return 1/(1+np.exp(-s))


def test(X):
  # forward propogate the calculations
  z1 = np.dot(X, w1)
  z2 = sigmoid(z1)
  z3 = np.dot(z2, w2)
  o = sigmoid(z3)
  o=np.around(o)
  return o



if __name__=='__main__':

  testing_data = np.loadtxt('test_input.txt', dtype=float)
  testing_data = np.hstack((testing_data, np.ones((testing_data.shape[0], 1), dtype=testing_data.dtype)))
  output = test(testing_data)
  print('Output is:\n', output)
