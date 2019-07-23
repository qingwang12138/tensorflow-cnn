# encoding = utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 导入基础数据集
mnist = input_data.read_data_sets('D:/reference/5-dataset/MNIST_data', one_hot=True)  # one_hot是指数组中只有一个元素是1，其他为0


# 定义函数
def computer_accuracy(v_xs, v_ys):  # 在计算test集时使用，传入测试集x，y的shape，也就是各个维度值，返回正确率
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})  # feed_dict传值列表，keep_prob为dropout变量值，返回prediction
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))  # 1指按行寻找，返回值是布尔值0、1
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # cast转换数据类型,求平均值就是求正确率
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})  # 第一个accuracy参数可以认为是指定运算到哪个tensor截止
    return result


def weight_variable(shape):  # 输入各维度值
    initial = tf.truncated_normal(shape, stddev=0.1)  # 产生随机初始权值
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  # 产生定值为0.1的矩阵
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')   # same的边缘是由0填充的，类似matlab中的full卷积


def max_pool_2x2(x):  # ksize为池化窗口大小
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 占位符，接受feed_dict传入的参数
xs = tf.placeholder(tf.float32, [None, 784])  # None指忽略该维度的值，这里由于仅给出占位符，无法不考虑batch大小，忽略batch值
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])  # -1指暂不考虑输入维度，1指图像通道数#print(x_image)==>[n_sample,28,28,1]

# CNN结构计算，所有层都包括权重、偏置、输出
# conv1 layer
W_conv1 = weight_variable([5, 5, 1, 32])  # 5x5指卷积大小，1为输入层特征层数，32为输出特征层数
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 因为采用的padding是same，长宽不变，为28，output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32

# conv2 layer
W_conv2 = weight_variable([5, 5, 32, 64])  # 5x5指卷积大小，32为输入层特征层数，64为输出特征层数
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

# func1 layer全连接层
W_fc1 = weight_variable([7*7*64, 1024])  # 1024为拉伸后的长度
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  # 将h_pool2的结果[n_samples,7,7,64]变为一维的[n_sample,7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # matmul矩阵相乘
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 随机抛掉一些神经元，防止过拟合，抛掉比例为keep_pro，本例为1，即不抛掉

# func2 layer全连接层
W_fc2 = weight_variable([1024, 10])  # 1024为拉伸后的长度
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # matmul矩阵相乘

# 计算误差，利用优化器自动调整权值和偏置
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))  # log是交叉熵，reduction_indices=[1]是指按照行维度求平均值
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# 初始化所有变量
sess.run(tf.global_variables_initializer())

# 输出
for i in range(500):
    batch_xs, batch_ys = mnist.train.next_batch(50)  # batch_xs/batch_ys为100x784，每次挑出100个x，y数据集，训练一次神经网络
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1})  # 图式计算开始，传入训练的每个参数，返回train_step
    if i % 10 == 0:  # 输入测试函数集，得出正确率
        print(computer_accuracy(batch_xs, batch_ys))