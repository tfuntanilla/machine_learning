import os
import random

import cv2
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

plt.style.use('seaborn')

accuracy_rates_pca = []
accuracy_rates_lda = []


def run_pca(X_train, X_test, n_dim):
    pca = PCA(n_components=n_dim)
    pca.fit(X_train.T)

    X_train_pca = pca.transform(X_train.T)
    X_test_pca = pca.transform(X_test.T)

    return X_train_pca, X_test_pca


def run_lda(X_train, y_train, X_test, n_dim):
    #  first use PCA to reduce the dimensionality of face images to d0 = 40
    d0 = 40
    pca = PCA(n_components=d0)
    pca.fit(X_train.T)

    X_train_ = pca.transform(X_train.T)
    X_test_ = pca.transform(X_test.T)

    # input of lda is the reduced-dim data from PCA
    lda = LinearDiscriminantAnalysis(n_components=n_dim)
    lda.fit(X_train_, y_train)

    X_train_lda = lda.transform(X_train_)
    X_test_lda = lda.transform(X_test_)

    return X_train_lda, X_test_lda


def classification(X_train, y_train, X_test, y_test, algo):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)

    y_pred = neigh.predict(X_test)

    correct_preds = accuracy_score(y_test, y_pred, normalize=False)
    accuracy_rate = accuracy_score(y_test, y_pred)

    print('-------------------------------------------------------------------------------------------------------')
    print(algo)
    print('-------------------------------------------------------------------------------------------------------')
    print('True = {}'.format(y_test))
    print('Predicted = {}'.format(y_pred))
    print('Total number of correct classifications = {}'.format(correct_preds))
    print('Accuracy rate = {}'.format(accuracy_rate))

    return y_pred, correct_preds


def run_experiments(n_components):
    total_correct_pca = 0
    total_test_cases_pca = 0
    total_correct_lda = 0
    total_test_cases_lda = 0

    # 20 independent experiments
    for exp_num in range(20):

        print('*******************************************************************************************************')
        print('Experiment #' + str(exp_num + 1))
        print('*******************************************************************************************************')

        # randomly choose 8 images per class to form the training set
        # the rest will be in the test set
        train_ims = []
        test_ims = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for y in range(8):
            choice = random.choice(test_ims)
            train_ims.append(choice)
            test_ims.remove(choice)
        print('Train images per class = ' + str(train_ims))
        print('Test images per class = ' + str(test_ims))

        x_train = []
        y_train = []
        x_test = []
        y_test = []

        # PREPROCESSING
        for i in range(1, 11):  # 10 subjects
            for j in range(1, 11):  # 10 images per subject
                # read the images as grayscale
                label = 's' + str(i)
                im_path = os.path.join('../hw2/att_faces_10', label, str(j) + '.pgm')
                im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
                im_vec = im.ravel()  # reshape to a vector of length 10304

                if j in train_ims:
                    x_train.append(im_vec)
                    y_train.append(label)
                else:
                    x_test.append(im_vec)
                    y_test.append(label)

        # stack training images of all 10 subjects
        X_train = np.transpose(np.array(x_train))
        X_test = np.transpose(np.array(x_test))

        ###########################
        # Run the algorithms
        ###########################
        X_train_pca, X_test_pca = run_pca(X_train, X_test, n_components)
        X_train_lda, X_test_lda = run_lda(X_train, y_train, X_test, n_components)

        ###########################
        # CLASSIFICATION FOR PCA
        ###########################
        y_pred_pca, correct_preds_pca = classification(X_train_pca, y_train, X_test_pca, y_test, 'PCA')
        total_correct_pca += correct_preds_pca
        total_test_cases_pca += len(y_pred_pca)

        ###########################
        # CLASSIFICATION FOR LDA
        ###########################
        y_pred_lda, correct_preds_lda = classification(X_train_lda, y_train, X_test_lda, y_test, 'LDA')
        total_correct_lda += correct_preds_lda
        total_test_cases_lda += len(y_pred_lda)

    ###########################
    # Gather the results
    ###########################
    accuracy_rate_pca = total_correct_pca / total_test_cases_pca
    accuracy_rates_pca.append(accuracy_rate_pca)

    accuracy_rate_lda = total_correct_lda / total_test_cases_lda
    accuracy_rates_lda.append(accuracy_rate_lda)


def plot_results():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    plt.plot(d, accuracy_rates_pca, linestyle='-', marker='o', color='c', label='PCA')
    plt.plot(d, accuracy_rates_lda, linestyle='-', marker='o', color='r', label='LDA')
    for i, j in zip(d, accuracy_rates_pca):
        ax.annotate(str(round(j, 3)), xy=(i, j))
    for i, j in zip(d, accuracy_rates_lda):
        ax.annotate(str(round(j, 3)), xy=(i, j))
    plt.xlabel('d')
    plt.ylabel('accuracy rate')
    plt.locator_params(axis='y', nbins=10)
    plt.locator_params(axis='x', nbins=50)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    d = [1, 2, 3, 6, 10, 20, 30]
    for dim in d:
        run_experiments(dim)

    print()
    print('d = ' + str(d))
    print('Accuracy rates PCA = ' + str(accuracy_rates_pca))
    print('Accuracy rates LDA = ' + str(accuracy_rates_lda))

    plot_results()
    print("Done!")
