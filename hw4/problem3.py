import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download dataset, no need to split since train and test sets are given separately
mnist = tf.keras.datasets.mnist
(x_traino, y_train), (x_testo, y_test) = mnist.load_data()

# reshape
x_train = np.reshape(x_traino, (60000, 28 * 28))
x_test = np.reshape(x_testo, (10000, 28 * 28))
x_train, x_test = x_train / 255.0, x_test / 255.0

# initialize logistic regression
logreg = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=100, verbose=2)

# fit and predict
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

# give accuracy report and show confusion matrix
print(classification_report(y_test, y_pred))
print('Recognition accuracy rate = ' + str(accuracy_score(y_test, y_pred)))
print(confusion_matrix(y_test, y_pred))
