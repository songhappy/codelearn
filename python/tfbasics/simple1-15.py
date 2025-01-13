import multiprocessing
import tensorflow as tf
print(tf.VERSION)

ncpu = multiprocessing.cpu_count()
config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu)
config.gpu_options.allow_growth = True
tf.Session(config=config).__enter__()

t = tf.constant(42.0)
u = tf.constant(37.0)
tu = tf.multiply(t, u)
ut = tf.multiply(u, t)
sess = tf.Session()
with sess.as_default():
   tu.eval()  # runs one step
   ut.eval()  # runs one step
   sess.run([tu, ut])  # evaluates both tensors in a single step

import tensorflow as tf

x = tf.placeholder("float", None)
y = x * 2
with tf.Session() as session:
    result1 = session.run(y, feed_dict={x: [1, 2, 3]})
    print(result1)

with tf.Session() as session:
    x_data = [[1, 2, 3],
              [4, 5, 6],]
    result = session.run(y, feed_dict={x: x_data})
    print(result)

assign =[]
for i in range(10):
    assign.append(tf.assign(i, i))
with tf.Session() as session:
    result = session.run(assign)
    print(result)