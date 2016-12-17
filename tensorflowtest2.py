from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)#load mnist data.
import tensorflow as tf
sess = tf.InteractiveSession()#import tensorflow.


x = tf.placeholder(tf.float32, shape=[None, 784])#these are flattened 28x28 input nodes.  \
y_ = tf.placeholder(tf.float32, shape=[None, 10])#these are 10 goals: correct outputs.     } these are included in mnist data set.

W = tf.Variable(tf.zeros([784,10])) #weights between input and output layers.
b = tf.Variable(tf.zeros([10]))     #biases for each weight.

sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b #calculation for output; y is output and now bound to weights, biases, and input.

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_)) #complicated operation but basically calculates error

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #minimize error

for i in xrange(500): #train
 batch = mnist.train.next_batch(100) #get next batch of 100 mnist characters
 train_step.run(feed_dict={x: batch[0], y_: batch[1]}) #assign inputs and expected outputs and trains


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) #applies softmax and counts correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #finds percentage of correct answers
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})) #prints
