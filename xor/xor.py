import numpy as np
import random

class Xor(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a.T)+b)
            a = a.T
        return a

    def GradientDescent(self, training_data, test_data, epochs, lr, output):
        n_test = len(test_data)
        for epoch in range(epochs):
            new_b = [np.zeros(b.shape) for b in self.biases]
            new_w = [np.zeros(w.shape) for w in self.weights]
            for x,y in zip(training_data, output):
                x = np.array(x)
                y = np.array(y)
                delta_new_b, delta_new_w = self.backprop(x, y)
                new_b = [nb+dnb for nb, dnb in zip(new_b, delta_new_b)]
                new_w = [nw+dnw for nw, dnw in zip(new_w, delta_new_w)]
            self.weights = [w-(lr/len(training_data))*nw
                            for w, nw in zip(self.weights, new_w)]
            self.biases = [b-(lr/len(training_data))*nb
                           for b, nb in zip(self.biases, new_b)]


    def backprop(self, x, y):
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activation = np.array(activation, ndmin=2)
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation.T)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            activation = activation.T


        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        delta = np.array(delta, ndmin=2)
        new_b[-1] = delta

        new_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            new_b[-l] = delta
            activations[-l-1] = np.array(activations[-l-1], ndmin=2)
            new_w[-l] = np.dot(delta, activations[-l-1])
        return (new_b, new_w)


    def test(self, test_data, y1):
        n = len(test_data)
        for i in test_data:
            print(i)
            i = np.array(i, ndmin=2)
            print(self.feedforward(i))




def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

f = open('train.txt', 'r')
data = f.readlines()
training_data = []
for row in data:
    training_data.append(row[:-2].strip().split(' '))

for row in training_data:
    for i in range(len(row)):
        row[i] = float(row[i])

y = []
for row in data:
    y.append(row[-2].strip().split(' '))

for row in y:
    for i in range(len(row)):
        row[i] = float(row[i])

g = open('test.txt', 'r')
data = g.readlines()
test_data = []
for row in data:
    test_data.append(row[:-2].strip().split(' '))

for row in test_data:
    for i in range(len(row)):
        row[i] = float(row[i])

y1 = []
for row in data:
    y1.append(row[-2].strip().split(' '))

for row in y1:
    for i in range(len(row)):
        row[i] = float(row[i])

xor = Xor([2, 3, 1])
print(xor.weights)
xor.GradientDescent(training_data, test_data, 5000, 0.006, y)
print(xor.weights)
print(xor.test(test_data, y1))
