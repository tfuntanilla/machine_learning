import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
plt.style.use('seaborn')

df = pd.read_excel('hw1/pima-indians-diabetes.xlsx')
no_diabetes = df.loc[df['Outcome'] == 0]
diabetes = df.loc[df['Outcome'] == 1]

n_values = [20, 40, 60, 80, 100]
accuracy_scores = []
for n in n_values:
    print('-------------------------------------------------------------------------------------')
    print('n=' + str(n))
    train_size = n
    test_size = (len(no_diabetes) - n) + (len(diabetes) - n)

    total_correct_pred = 0
    for i in range(0, 1000):
        train_no_diabates, test_no_diabates = \
            train_test_split(no_diabetes, train_size=n, test_size=int(len(no_diabetes) - n))
        train_diabates, test_diabates = train_test_split(diabetes, train_size=n, test_size=int(len(diabetes) - n))

        train = pd.concat([train_no_diabates, train_diabates], ignore_index=True)
        test = pd.concat([test_no_diabates, test_diabates], ignore_index=True)

        X_train = train.loc[:, train.columns != 'Outcome'].values
        y_train = train.loc[:, train.columns == 'Outcome'].values

        X_test = test.loc[:, test.columns != 'Outcome'].values
        y_test = test.loc[:, test.columns == 'Outcome'].values

        reg = LinearRegression().fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        class_pred = []
        for t_hat in y_pred:
            if t_hat >= 0.5:
                class_pred.append(1)
            else:
                class_pred.append(0)

        total_correct_pred += accuracy_score(y_test, class_pred, normalize=False)
        exp_correct_pred = accuracy_score(y_test, class_pred, normalize=False)
        exp_accuracy = accuracy_score(y_test, class_pred)
        print('Experiment #' + str(i) +
              ', correct predictions=' + str(exp_correct_pred) + '/' + str(len(test)) +
              ', accuracy=' + str(exp_accuracy))

    # after 1000 experiments
    accuracy = total_correct_pred / (test_size * 1000)
    accuracy_scores.append(accuracy)

    print('Accuracy=', accuracy)
    print()

plt.plot(n_values, accuracy_scores, 'o')
plt.xlabel('n')
plt.ylabel('accuracy score')
plt.locator_params(axis='y', nbins=10)
plt.locator_params(axis='x', nbins=7)
plt.show()
