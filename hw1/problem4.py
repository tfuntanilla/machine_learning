from copy import deepcopy

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

plt.style.use('seaborn')
from math import pow

df = pd.read_excel('hw1/Iris.xls')
data = df.iloc[:, [1, 2, 3, 4]].values  # sepal length, sepal width, petal length, and petal width
category = df.iloc[:, 5].values  # outcome

k = 3  # number of clusters
n = data.shape[0]  # number of training data
num_of_features = data.shape[1]  # number of features in the data

# randomly choose three samples as the initial cluster centers
centers = data[np.random.randint(0, data.shape[0], 3)]
print("Randomly chosen centers = \n" + str(centers))

# Plot the data and the centers generated as random
colors = ['red', 'blue', 'green']
# for i in range(n):
#     plt.scatter(data[i, 0], data[i, 1], s=7, color=colors[int(category[i])])
# plt.scatter(centers[:, 0], centers[:, 1], marker='*', c='g', s=150)

centers_old = np.zeros(centers.shape)  # to store old centers
centers_new = deepcopy(centers)  # store new centers

clusters = np.zeros(n)
distances = np.zeros((n, k))

iter = 1
obj_func_value = np.linalg.norm(centers_new - centers_old)
print('Iteration #', iter)
print('Objective function value J =', obj_func_value)
print()

iter_array = []
iter_array.append(iter)
obj_func_array = []
obj_func_array.append(obj_func_value)

while obj_func_value >= pow(10, -5):
    # measure the distance to every center
    for i in range(k):
        distances[:, i] = np.linalg.norm(data - centers[i], axis=1)
    # assign all training data to closest center
    clusters = np.argmin(distances, axis=1)

    centers_old = deepcopy(centers_new)
    # calculate mean for every cluster and update the center
    for i in range(k):
        centers_new[i] = np.mean(data[clusters == i], axis=0)
    obj_func_value = np.linalg.norm(centers_new - centers_old)

    iter += 1
    print('Iteration', iter)
    print('Objective function value J =', obj_func_value)
    print()

    obj_func_array.append(obj_func_value)
    iter_array.append(iter)

fig1 = plt.figure()
# plot the clusters
colors = ['red', 'blue', 'green']
for i in range(n):
    plt.scatter(data[i, 0], data[i, 1], s=7, color=colors[int(category[i])])

# plot the objective function value J versus the iteration number Iter
fig2 = plt.figure()
ax = fig2.add_subplot(111)
plt.plot(iter_array, obj_func_array, 'o')
for i, j in zip(iter_array, obj_func_array):
    ax.annotate(str(round(j, 3)), xy=(i, j))
plt.xlabel('iteration number')
plt.ylabel('objective function value J')
plt.locator_params(axis='x', nbins=3)
plt.show()
