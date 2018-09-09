import numpy as np
import random

dataset = [
    [0, 0, 0, 0],
	[0, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 1]
]

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def activation(x):
    if sigmoid(x) >= 0.5:
        return 1
    else:
        return 0

def checkAccuracy(test_data, weights):
    c = 0;
    for row in test_data:
        wt_sum = 0.0
        wt_sum += weights[0]
        for i in range(len(row)-1):
            wt_sum += weights[i+1]*row[i]
        prediction = activation(wt_sum)
        if prediction == row[-1]:
            c += 1

    accuracy = ((c * 100)/len(test_data))
    return accuracy

def perceptron(dataset):
    weights = [np.random.uniform(-1, 1) for i in range(len(dataset[0]))]
    lr = 0.1
    epochs = 100
    sum_error = 0.0
    for epoch in range(epochs):
        random.shuffle(dataset)
        for row in dataset:
            w_sum = 0.0;
            w_sum += weights[0]
            for i in range(len(row)-1):
                w_sum += weights[i+1]*row[i]
            prediction = activation(w_sum)
            error = row[-1] - prediction
            sum_error += error
            weights[0] += (lr * error)
            for i in range(len(row)-1):
                weights[i+1] += (lr * error * row[i])
        if sum_error == 0.0:
            print("TRAINED")
            break
        print("epoch = %d, error = %.2f" % (epoch, sum_error))
    return weights

weights = perceptron(dataset)

print(weights)

f = open('test.txt', 'r')

data = f.readlines()

test_data = []

for row in data:
    test_data.append(row[:-1].strip().split(' '))

for row in test_data:
    for i in range(len(row)):
        row[i] = float(row[i])

random.shuffle(test_data)

accuracy = checkAccuracy(test_data, weights)
print("Accuracy = %.2f%%\n" % (accuracy))

print("TEST BY INPUT\n")

while True:
    wt = input().strip().split(' ')

    for i in range(len(wt)):
        wt[i] = float(wt[i])

    if wt[0] == -1 and len(wt) == 1:
        break

    wt_sum = 0.0
    wt_sum += weights[0]

    for i in range(len(dataset[0])-1):
        wt_sum += weights[i+1] * wt[i]

    print(activation(wt_sum))
