
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1,28,28,1])
x_paints = bias_variable([1,28,28,1])


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

PAINT_1 = max_pool_2x2(tf.nn.relu(conv2d(x_paints, W_conv1) + b_conv1))


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


PAINT_2 = max_pool_2x2(tf.nn.relu(conv2d(PAINT_1, W_conv2) + b_conv2))


W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

PAINT_3 = tf.nn.relu(tf.matmul(tf.reshape(PAINT_2, [-1, 7*7*64]), W_fc1) + b_fc1)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# saver = tf.train.Saver({'l1': W_fc1, 'l2': W_fc2,'b1': b_fc1, 'b2': b_fc2})

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

PAINT_4 = tf.matmul(PAINT_3, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

paint_step = tf.train.AdamOptimizer(1e-4).minimize(-PAINT_4[0][0])


sess.run(tf.global_variables_initializer())


for i in range(5000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


for i in range(20000):
  if i%100 == 0:
    paint_accuracy = PAINT_4.eval()
    print("step %d, training accuracy %g"%(i, paint_accuracy[0][0]))
  paint_step.run()

# savepath = saver.save(sess, 'trained-mnist')
# del saver

# tf.reset_default_graph()
# sess = tf.InteractiveSession()

# x = bias_variable([784])

# x_image = tf.reshape(x, [-1,28,28,1])

# W_conv1 = weight_variable([5, 5, 1, 32])
# b_conv1 = bias_variable([32])

# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)


# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])

# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)



# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])

# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob = tf.placeholder(tf.float32)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])

# saver = tf.train.Saver({'l1': W_fc1, 'l2': W_fc2,'b1': b_fc1, 'b2': b_fc2})

# y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# paint_step = tf.train.AdamOptimizer(1e-4).minimize(-y_conv[0])

# saver.restore(sess,savepath)

# for i in range(20000):
#   if i%100 == 0:
#     train_accuracy = y_conv.eval(feed_dict={keep_prob: 1.0})
#     print("step %d, training accuracy %g"%(i, train_accuracy))
#   paint_step.run(feed_dict={keep_prob: 0.5},var_list={x})





