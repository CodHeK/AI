#perceptron
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x));

def predict(x, weights):
    #hypothesis
    h = weights[0] + weights[1] * (x)
    threshold = sigmoid(5000)
    if sigmoid(h) >= threshold:
        return 1.0
    else:
        return 0.0


def train(train_data, l_rate, epochs):
    #random weights to initalize
    weights = [np.random.uniform(-0.5, 0.5) for i in range(len(train_data[0]))]
    for epoch in range(epochs):
        cost = 0.0
        for each_row in train_data:
            predicted_val = predict(each_row[0], weights)
            error = each_row[-1] - predicted_val
            cost += error**2
            weights[0] += (l_rate * error) #adjusting bias weight
            weights[1] += (l_rate * error * each_row[0]) #adjusting weights for the input feature
        cost = cost/(2*len(train_data))
        if cost == 0.0:
            print("*** FULLY TRAINED ***\n")
            break
        print("epoch = %d, error = %0.3f" % (epoch, cost))
    return weights, epoch


def test(test_data, weights):
    predicted_vals = []
    actual_vals_for_test_data = []
    for x in test_data:
        predicted_vals.append(predict(x, weights))
        if x < 5000:
            actual_vals_for_test_data.append(0.0)
        else:
            actual_vals_for_test_data.append(1.0)

    c = 0
    total = len(predicted_vals)
    for i in range(total):
        if predicted_vals[i] == actual_vals_for_test_data[i]:
            c += 1

    accuracy = (c*100)/total

    return accuracy


#prepare the dataset
rands = np.random.randint(1000, 10000, size=1000)

train_data = []
test_data = np.random.randint(1000, 10000, size=2500)

for i in rands:
    if i < 5000:
        train_data.append([i, 0.0])
    else:
        train_data.append([i, 1.0])

l_rate = np.random.uniform(0.0005, 0.0015, size=5)

lr = 0.0006
epoch = 1000

weights, epoch = train(train_data, lr, epoch)

#Finding optimal learning rate first

# min_epoch = 100000
#
# for lr in l_rate:
#     weights, epoch = train(train_data, lr, epoch)
#     if epoch < min_epoch:
#         min_epoch = epoch
#         optimal_lr = lr
#
# print("Optimal Value of Learning rate = %.4f finishes training in %d epochs\n" % (optimal_lr, min_epoch))
#
# weights, epoch = train(train_data, optimal_lr, min_epoch)

#Testing
print("Hypothesis function -> H = %.3f + %.3f * (X)\n" % (weights[0], weights[1]))

print("Accuracy = %.2f%%\n" % (test(test_data, weights)))

print("Training and Testing is Complete! \n")

while True:
    new_value = input("Input the value of house to check which class it belongs to! (ENTER -1 to EXIT) \n")
    if int(new_value) == -1:
        break
    if(predict(int(new_value), weights) == 0.0):
        hclass = "A"
    else:
        hclass = "B"
    print("Your house belongs to Class %s \n" % (hclass))
