# Simple multiple linear regression
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_boston

# Data
dataset = load_boston()
X_data = dataset.data.astype(np.float32)
y_data = dataset.target.astype(np.float32)

# Creating the model
w = tf.Variable(tf.zeros([X_data.shape[1], 1]))
b = tf.Variable(np.array([y_data.min()]).astype(np.float32))
y = tf.add(tf.matmul(X_data, w), b)

# Learning rate
alpha = 0.0001
# Cost function
loss_function = tf.reduce_mean(tf.square(tf.log(y)-tf.log(y_data)))
#cost = tf.reduce_mean(tf.square(y-y_data))
# Optimization
optimizer = tf.train.GradientDescentOptimizer(alpha)
# Minimize the cost
train = optimizer.minimize(loss_function)

init = tf.global_variables_initializer()
with tf.Session() as sess:

    # Initializing the variables
    sess.run(init)

    for epoch in range(1000):
        _, loss = sess.run([train, loss_function])