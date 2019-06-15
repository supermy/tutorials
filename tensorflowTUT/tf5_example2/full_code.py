# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

# create data 创建向量（一维数组；二维数组是矩阵；三位以上是张量）
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3 #向量运算结果也是向量，向量的枚举加法与乘法运算


### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) #权重;random_uniform,均匀分布随机数:形状，最小值，最大值；
biases = tf.Variable(tf.zeros([1])) #斜度;zeros,创建一个所有元素都设置为零的张量.

y = Weights*x_data + biases

# tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
loss = tf.reduce_mean(tf.square(y-y_data)) #损失函数：最小二乘法;tf.square 平方;

optimizer = tf.train.GradientDescentOptimizer(0.5)#构造一个新的梯度下降优化器实例;0.5,优化器将采用的学习速率;
train = optimizer.minimize(loss)  #在这里rate为0.01,因为这个示例也是多维函数，所以也要用到偏导数来进行逐步向最优解靠近。


### create tensorflow structure end ###

sess = tf.Session()
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()


#session 是会话，是运行的并行避免干扰（图计算还有数据的并行）；
sess.run(init)


for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))


