import numpy as np

threshold = 0.2

def activation_fn(sum):
    ''' Step Function '''
    if (sum > 0):
        return 1
    return 0


def predict(data, weights):
    ''' Calculate the weighted sum '''
    weighted_sum = weights.T.dot(data) - threshold
    a = activation_fn(weighted_sum)
    return a

def test(data, weights):
    ''' Loop through testing set and print the output'''
    for i in range(data.shape[0]):
        output = predict(data[i], weights)
        print(output)

if __name__=='__main__':

    testing_data = np.loadtxt('test_input.txt')
    weights = np.loadtxt('corrected_weights.txt')

    test(testing_data, weights)