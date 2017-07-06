import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 6)

# Creating train data
X_data = np.random.rand(100).astype(np.float32)
y_data =  5*X_data+6
# Adding gaussian noise to y_data
y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.1))(y_data)


# Creating the model
m = tf.Variable(np.random.uniform(0.0,1.0))
b = tf.Variable(np.random.uniform(0.0, 1.0))
y = tf.add(tf.multiply(X_data, m), b)

# Learning rate
alpha = 0.1
# Cost function
cost = tf.reduce_mean(tf.square(y-y_data))
# Optimization
optimizer = tf.train.GradientDescentOptimizer(alpha)
# Minimize the cost
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:

    # Initializing the variables
    sess.run(init)

    for epoch in range(100):
        sess.run([train, m, b])

    plt.plot(X_data, y_data, 'ro', label='Original data')
    plt.plot(X_data, sess.run(m) * X_data + sess.run(b), label='Fitted line')

    plt.legend()
    plt.show()