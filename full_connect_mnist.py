# 使用全连接的mnist识别
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import data.input_data as input_data
mnist = input_data.read_data_sets('D:/reference/5-dataset/MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, [None,784])
y_ = tf.placeholder(tf.float32,[None,10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W)+b)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=tf.log(y)))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
for i in range(1000):
    if i % 50 == 0:
        print(i)
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_x,y_:batch_y})
correct_prediction = tf.equal(tf.arg_max(y, 1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print(sess.run(accuracy,feed_dict = {x:mnist.test.images,y_:mnist.test.labels}))
sess.close()