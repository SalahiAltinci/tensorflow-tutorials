# Based on Cognitive class AI tutorial

import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
X_data, y_data = iris.data[:-1,:], iris.target[:-1]
y_data= pd.get_dummies(y_data).values
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30, random_state=42)

num_features = X_data.shape[1]
num_labels = y_data.shape[1]

print("num_labels", num_labels)
print("num_features", num_features)

# * 'None' means TensorFlow shouldn't expect a fixed number in that dimension
X = tf.placeholder(tf.float32, [None, num_features])
y = tf.placeholder(tf.float32, [None, num_labels])

w = tf.Variable(tf.random_normal([num_features, num_labels],
                                       mean=0,
                                       stddev=0.01,
                                       name="weights"))

b = tf.Variable(tf.random_normal([1, num_labels],
                                    mean=0,
                                    stddev=0.01,
                                    name="bias"))

logits = tf.matmul(X, w) + b
activation = tf.nn.sigmoid(logits, name="activation")
cost_function = tf.nn.l2_loss(activation-y )

# Training
num_epochs = 1000
learning_rate = 0.0008

training = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
prediction_function = tf.equal(tf.argmax(activation, 1),tf.argmax(y, 1))
accuracy_function = tf.reduce_mean(tf.cast(prediction_function, "float"))

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    epoch_values = []
    accuracy_values = []
    cost_values = []

    for epoch in range(num_epochs):
        step = session.run(training, feed_dict={X: X_train, y: y_train})
        if epoch % 100 == 0:
            accuracy, cost = session.run([accuracy_function, cost_function], feed_dict={X: X_train, y: y_train})

            # Cost and accuracy history
            epoch_values.append(epoch)
            accuracy_values.append(accuracy)
            cost_values.append(cost)

    test_accuracy = session.run(accuracy_function, feed_dict={X: X_test, y: y_test})
    print("Accuracy on test set: %s" %test_accuracy)

    plt.plot(accuracy_values)
    plt.show()