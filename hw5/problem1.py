import matplotlib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

plt.style.use('seaborn')

# Download dataset, no need to split since train and test sets are given separately
mnist = tf.keras.datasets.mnist
(x_traino, y_train), (x_testo, y_test) = mnist.load_data()

# Reshape
x_train = np.reshape(x_traino, (60000, 28 * 28))
x_test = np.reshape(x_testo, (10000, 28 * 28))
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create the two-layer neural network
model = tf.keras.models.Sequential()
# The hidden layer has 512 node, and adopts the ReLU activation function
# Need to also pass the input shape here
model.add(tf.keras.layers.Dense(512, input_shape=(x_train.shape[1],), activation='relu'))
# the output layer has 10 nodes and adopts the softmax activation function
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# Use the cross-entropy error function - targets are integers so use sparse_categorical_crossentropy
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Run 5 epochs
model.fit(x_train, y_train, epochs=5)

# Check score of model
scores = model.evaluate(x_train, y_train)
print("\nAccuracy on train set: %.2f%%" % (scores[1] * 100))

# Predict
y_pred = model.predict_classes(x_test)

# Give accuracy report and show confusion matrix
print(classification_report(y_test, y_pred))
print('Recognition accuracy rate = ' + str(accuracy_score(y_test, y_pred)))

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix
classes = range(10)
fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap(name='GnBu'))
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       ylabel='True label',
       xlabel='Predicted label')

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.grid('off')

fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.show()
