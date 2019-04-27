import os

import cv2
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

plt.style.use('seaborn')

train_ims = [1, 3, 4, 5, 7, 9]
test_ims = [2, 6, 8, 10]

x_train = []
y_train = []
x_test = []
y_test = []

# PREPROCESSING
for i in range(1, 11):  # 10 subjects
    for j in range(1, 11):  # 10 images per subject
        # read the images as grayscale
        label = 's' + str(i)
        im_path = os.path.join('./att_faces_10', label, str(j) + '.pgm')
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        im_vec = im.ravel()  # reshape to a vector of length 10304

        if j in train_ims:
            x_train.append(im_vec)
            y_train.append(label)
        else:
            x_test.append(im_vec)
            y_test.append(label)

# stack 6 training images of all 10 subjects
X_train = np.transpose(np.array(x_train))

# zero center
mu_train = (np.dot(X_train, np.identity(X_train.shape[1]))) / X_train.shape[1]
X_train = X_train - mu_train

# print('X_train = ')
# print(X_train)
# print('Shape of X_train = {}'.format(X_train.shape))

# do same processing for test
X_test = np.transpose(np.array(x_test))
mu_test = (np.dot(X_test, np.identity(X_test.shape[1]))) / X_test.shape[1]
X_test = X_test - mu_test

# print('X_test = ')
# print(X_test)
# print('Shape of X_test = {}'.format(X_test.shape))

# PCA AND CLASSIFICATION
K = [1, 2, 3, 6, 10, 20, 30, 50]
im_height = 112
im_width = 92

accuracy_rates = []  # stores the accuracy rates for the different K values

for k in K:
    #################
    # PCA
    #################
    pca = PCA(n_components=k)
    pca.fit(X_train.T)
    # print('Rank-{}'.format(k) + ' principal components = ')
    # print(pca.components_)

    # project the data on the principal components
    X_train_pca = pca.transform(X_train.T)
    X_test_pca = pca.transform(X_test.T)

    #################
    # CLASSIFICATION
    #################
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train_pca, y_train)

    y_pred = neigh.predict(X_test_pca)

    correct_preds = accuracy_score(y_test, y_pred, normalize=False)
    accuracy_rate = accuracy_score(y_test, y_pred)

    print('K = {}'.format(k))
    print('True = {}'.format(y_test))
    print('Predicted = {}'.format(y_pred))
    print('Total number of correct classifications = {}'.format(correct_preds))
    print('Accuracy rate = {}'.format(accuracy_rate))

    accuracy_rates.append(accuracy_rate)

# PLOT
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(K, accuracy_rates, linestyle='-', marker='o')
for i, j in zip(K, accuracy_rates):
    ax.annotate(str(round(j, 3)), xy=(i, j))
plt.xlabel('K')
plt.ylabel('accuracy rate')
plt.locator_params(axis='y', nbins=10)
plt.locator_params(axis='x', nbins=50)
plt.show()
