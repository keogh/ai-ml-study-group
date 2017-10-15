import numpy as np
import matplotlib.pyplot as plt
from random import choice
import math

b = np.random.rand(3) # weights or betas
alpha = 0.1 #learning rate
n = 100 # training iterations


def plot_data(set1, set2):
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    x, y = set1.T
    group_1 = plt.scatter(x, y, c="red")
    x2, y2 = set2.T
    group_2 = plt.scatter(x2, y2, c="blue")
    plt.legend([group_1, group_2], ['Iris-setosa', 'Iris-otras'])
    plt.show()

def get_iris_data():
    iris_data = np.genfromtxt('iris.data', delimiter=',', usecols=(1,2))
    iris_1 = np.array(iris_data[:50]) # Iris-setosa
    iris_2 = np.array(iris_data[50:]) # Iris-otra
    # plot_data(iris_1, iris_2)
    np.random.shuffle(iris_1)
    np.random.shuffle(iris_2)
    training_data_iris = []
    validation_data_iris=[]
    n = 0
    for item in iris_1:

        if n > 35:
            validation_data_iris.append([np.append([1], item), 0])
        else:
            training_data_iris.append([np.append([1], item), 0])
        n+=1
    n = 0
    for item in iris_2:
        if n > 60:
            validation_data_iris.append([np.append([1], item), 1])
        else:
            training_data_iris.append([np.append([1], item), 1])
        n+=1
    np.random.shuffle(training_data_iris)
    np.random.shuffle(validation_data_iris)
    return training_data_iris, validation_data_iris

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# x: input vector
# y: expected result
# h: h thetha, sigmoid result
def adjust_weights(x, y, h):
    global b, alpha
    b = b - (alpha * (h - y) * x)

def train(training_data):
    for i in range(n):
        x, expected = choice(training_data)
        z = np.dot(b, x)
        result = sigmoid(z)
        error = expected - result
        adjust_weights(x, expected, result)

def validate(data):
    dots_red = []
    dots_blue = []
    for x, answer in data:
        z = np.dot(b, x)
        result = sigmoid(z)
        classification = 1 if result >= 0.5 else 0
        print("{}: {} -> {} -> {}".format(x[1:3], result, classification, answer))
        if classification > 0:
            dots_blue.append(x[1:3])
        else:
            dots_red.append(x[1:3])
    return dots_blue, dots_red

def main():
    # get data
    # plot data
    # select train data from data
    training_data, validation_data = get_iris_data()
    
    # train
    # TODO: plot taining
    train(training_data)
    
    # validate/predict
    dots_blue, dots_red = validate(validation_data)

    # plot predictions and divider rect
    x3, y3 = np.array(dots_red).T
    group_1 = plt.scatter(x3, y3, c="red")
    x4, y4 = np.array(dots_blue).T
    group_2 = plt.scatter(x4, y4, c="blue")
    plt.legend([group_1, group_2], ['Iris-setosa', 'Iris-otras'])

    # Draw the perceptron line
    # b0+b1x+b2y=0
    # y = (-b0-b1x)/b2
    line_x1 = 0
    line_y1 = -b[0]/b[2]
    line_x2 = 8
    line_y2 = (-b[0] - b[1]*line_x2)/b[2]
    plt.plot([line_x1, line_x2], [line_y1, line_y2])
    plt.show()

if __name__ == '__main__':
    main()
