import numpy as np

class Perceptron(object):

    def __init__(self, input, output):
        ''' Class Initialization '''
        self.input = input
        self.output = output
        self.shape = input.shape
        self.weights = np.random.random((self.shape[1],)) - 0.5

    def activation_fn(self, sum):
        ''' Step Function '''
        if (sum > 0):
            return 1
        return 0

    def predict(self, x, threshold):
        ''' Calculate the weighted sum '''
        weighted_sum = self.weights.T.dot(x) - threshold
        a = self.activation_fn(weighted_sum)
        return a

    def train(self, threshold, lr, epoch):
        ''' Loop through training set and train the dataset'''
        print(self.weights)

        for _ in range(epoch):
            for i in range(self.shape[0]):
                y = self.predict(self.input[i], threshold)
                e = self.output[i] - y
                self.weights += e * lr * self.input[i]
                print(self.weights)


if __name__=='__main__':

    training_data = np.loadtxt('training_input.txt')
    expec_output = np.loadtxt('training_output.txt', int)

    lr = 0.1                # learning rate
    threshold = 0.2         # threshold
    epoch = 100             # epoch

    and_func = Perceptron(training_data, expec_output)
    and_func.train(threshold, lr, epoch)
    np.savetxt('corrected_weights.txt', and_func.weights)