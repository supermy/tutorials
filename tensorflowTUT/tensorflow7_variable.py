# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.

Guava Graph 参见核心部件

"""
from __future__ import print_function
import tensorflow as tf

state = tf.Variable(0, name='counter')   # 创建一个状态节点，类似于全局变量；

#print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one) #定义操作
update = tf.assign(state, new_value)  #定义函数 把 add_operation 的结果赋值给 var

# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

#图的并行运算包括计算的并行与数据的并行；

#在会话里面执行，可以并行运算，互不干扰；
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3): #类似于递归；var 相当于是状态节点，在计算中进行传递；
        sess.run(update) #运行函数，var 的值每次都更新；0+1；1+1；2+1
        print(sess.run(state))

