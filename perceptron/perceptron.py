from random import choice
from numpy import array, dot, random, genfromtxt, append
from pylab import plot, ylim, show
import matplotlib.pyplot as plt

unit_step = lambda x: 1 if x >= 0 else -1

training_data_OR = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 1),
    (array([1,0,1]), 1),
    (array([1,1,1]), 1),
]

training_data_AND = [ (array([0,0,1]), 0),
    (array([0,1,1]), 0),
    (array([1,0,1]), 0),
    (array([1,1,1]), 1),
]

iris_data = genfromtxt('iris.data', delimiter=',', usecols=(1,2))
iris_1 = array(iris_data[:50]) # Iris-setosa
iris_2 = array(iris_data[50:]) # Iris-otra

plt.xlabel('sepal length')
plt.ylabel('sepal width')

x, y = iris_1.T
group_1 = plt.scatter(x, y, c="red")

x2, y2 = iris_2.T
group_2 = plt.scatter(x2, y2, c="blue")

plt.legend([group_1, group_2], ['Iris-setosa', 'Iris-otras'])
plt.show()

random.shuffle(iris_1)
random.shuffle(iris_2)
training_data_iris = []
validation_data_iris=[]
n = 0
for item in iris_1:
    if n > 35:
        validation_data_iris.append([append(item, [1], axis=1), -1])
    else:
        training_data_iris.append([append(item, [1], axis=1), -1])
    n+=1
n = 0
for item in iris_2:
    if n > 60:
        validation_data_iris.append([append(item, [1], axis=1), 1])
    else:
        training_data_iris.append([append(item, [1], axis=1), 1])
    n+=1
random.shuffle(training_data_iris)
random.shuffle(validation_data_iris)
# print(validation_data_iris)
# exit()

training_data = training_data_iris
validation_data = validation_data_iris

w = random.rand(3)
errors = []
eta = 0.2
n = 100

for i in range(n):
    x, expected = choice(training_data)
    result = dot(w, x)
    error = expected - unit_step(result)
    errors.append(error)
    w += eta * error * x

# for x, _ in training_data:
#     result = dot(x, w)
#     print("{}: {} -> {}".format(x[:2], result, unit_step(result)))

dots_red = [];
dots_blue = [];
for x, answer in validation_data:
    result = dot(x, w)
    classification = unit_step(result);
    print("{}: {} -> {} -> {}".format(x[:2], result, classification, answer))
    if classification > 0:
        dots_blue.append(x[:2])
    else:
        dots_red.append(x[:2])

ylim([-1,1])
plot(errors)
show()

plt.ylim([0, 8])
plt.xlim([0, 5])
x3, y3 = array(dots_red).T
group_1 = plt.scatter(x3, y3, c="red")

x4, y4 = array(dots_blue).T
group_2 = plt.scatter(x4, y4, c="blue")

plt.legend([group_1, group_2], ['Iris-setosa', 'Iris-otras'])

# Draw the perceptron line
# w0x+w1y+w2=0
line_x1 = 0
line_y1 = -w[2]/w[1]
line_x2 = 5
line_y2 = (-w[0]*line_x2 - w[2])/w[1]
plt.plot([line_x1, line_x2], [line_y1, line_y2])

plt.show()
