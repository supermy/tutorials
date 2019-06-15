# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1, m2)
            #3*2+3*2

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
print(result.shape)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

print("=======================================================")

a = tf.constant([[1,2],
                 [3,4]])
b = tf.constant([[0,0],
                 [1,0]])
c =a *b  #逐个相同位置元素相乘
d = tf.matmul(a,b) #矩阵相乘，矩阵转置；
        #1*0+2*1,1*0+2*0 = 2,0
        #3*0+4*1,3*0+4*0 = 4,0
with tf.Session() as sess:
    print(sess.run(a))
    print("")
    print(sess.run(b))
    print("")
    print(sess.run(c))
    print("")
    print(sess.run(d))


