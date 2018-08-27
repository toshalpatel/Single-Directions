from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 128
n_classes = 10

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  return f.truncated_normal(shape, stddev=0.1)

def bias_variable(shape):
  return tf.constant(0.1, shape=shape)

def conv2d(x, W, stride):
  return tf.layers.conv2d(x, W, strides=stride, kernel_size=(3,3), padding="same", activation=tf.nn.relu)

def max_pool_2d(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], padding='same')

def convolutional_nn(x):
    weights = { 'W_conv1': tf.Variable(tf.random_normal([5,5,1,64])),
                'W_conv2': tf.Variable(tf.random_normal([5,5,64,64])),
                'W_conv3': tf.Variable(tf.random_normal([5,5,64,128])),
                'W_conv4': tf.Variable(tf.random_normal([5,5,128,128])),
                'W_conv5': tf.Variable(tf.random_normal([5,5,128,128])),
                'W_conv6': tf.Variable(tf.random_normal([5,5,128,256])),
                'W_conv7': tf.Variable(tf.random_normal([5,5,256,256])),
                'W_conv8': tf.Variable(tf.random_normal([5,5,256,256])),
                'W_conv9': tf.Variable(tf.random_normal([5,5,256,512])),
                'W_conv10': tf.Variable(tf.random_normal([5,5,512,512])),
                'W_conv11': tf.Variable(tf.random_normal([5,5,512,512])),
                'W_fc': tf.Variable(tf.random_normal([7*7*512*1024])),
                'out': tf.Variable(tf.random_normal([512,n_classes])),}

    biases = {'b_conv1': tf.Variable(tf.random_normal([64])),
                'b_conv2': tf.Variable(tf.random_normal([64])),
                'b_conv3': tf.Variable(tf.random_normal([128])),
                'b_conv4': tf.Variable(tf.random_normal([128])),
                'b_conv5': tf.Variable(tf.random_normal([128])),
                'b_conv6': tf.Variable(tf.random_normal([256])),
                'b_conv7': tf.Variable(tf.random_normal([256])),
                'b_conv8': tf.Variable(tf.random_normal([256])),
                'b_conv9': tf.Variable(tf.random_normal([512])),
                'b_conv10': tf.Variable(tf.random_normal([512])),
                'b_conv11': tf.Variable(tf.random_normal([512])),
                'b_fc': tf.Variable(tf.random_normal([1024])),
                'out': tf.Variable(tf.random_normal([n_classes])),}

    x = tf.reshape(x, shape=[+1, 28, 28, 1])

    conv1 = conv2d(x, weights['W_conv1'], 1)
    conv1 = max_pool_2d(conv1)
    conv2 = conv2d(x, weights['W_conv2'], 1)
    conv2 = max_pool_2d(conv2)
    conv3 = conv2d(x, weights['W_conv3'], 2)
    conv3 = max_pool_2d(conv3)
    conv4 = conv2d(x, weights['W_conv4'], 1)
    conv4 = max_pool_2d(conv4)
    conv5 = conv2d(x, weights['W_conv5'], 1)
    conv5 = max_pool_2d(conv5)
    conv6 = conv2d(x, weights['W_conv6'], 2)
    conv6 = max_pool_2d(conv6)
    conv7 = conv2d(x, weights['W_conv7'], 1)
    conv7 = max_pool_2d(conv7)
    conv8 = conv2d(x, weights['W_conv8'], 1)
    conv8 = max_pool_2d(conv8)
    conv9 = conv2d(x, weights['W_conv9'], 2)
    conv9 = max_pool_2d(conv9)
    conv10 = conv2d(x, weights['W_conv10'], 1)
    conv10 = max_pool_2d(conv10)
    conv11 = conv2d(x, weights['W_conv11'], 1)
    conv11 = max_pool_2d(conv11)
    fc = tf.reshape(conv2, [+1, 7*7*512])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])

    output = tf.matmul(fc, weights['out']+biases['out'])

    return output

def train_nn(x):
    prediction = convolutional_nn(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer = tf.train.GradientDescentOptimizer(cost)

    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                                Y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

train_nn(x)