# Import useful libraries.
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

ROOT_DIR = os.path.abspath("../")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PATH_D = os.path.join(ROOT_DIR,"deprecated/MNIST_data/")
# Import dataset with one-hot encoding of the class labels.
def get_data():
  return input_data.read_data_sets(PATH_D, one_hot=True)

# Placeholders to feed train and test data into the graph.
# Since batch dimension is 'None', we can reuse them both for train and eval.
def get_placeholders():
  x = tf.placeholder(tf.float32, [None, 784])
  y_ = tf.placeholder(tf.float32, [None, 10])
  return x, y_

log_period_updates =200
batch_size = 100
settings = [(10000000000000, 0.0001), (1000000000, 0.005), (100000000, 0.1), (100000000, 0.001), (100000000, 0.003), (100000000, 0.004), (1000000000, 0.004) ,(1000000000, 0.006) ,(1000000000, 0.007) ]



def main_func():
    print('Training Model 3')
    # Train Model 1 with the different hyper-parameter settings.
    for (num_epochs, learning_rate) in settings:

        # Reset graph, recreate placeholders and dataset.
        tf.reset_default_graph()  # reset the tensorflow graph
        x, y_ = get_placeholders()
        mnist = get_data()  # use for training.
        eval_mnist = get_data()  # use for evaluation.

        #####################################################
        # Define model, loss, update and evaluation metric. #
        #####################################################
        # Construct the initialiser objects
        initializer = tf.contrib.layers.xavier_initializer()
        zeros_initializer = tf.zeros_initializer()

        # Initialize first layer weigths
        W1 = tf.Variable(initializer([784, 32]))
        b1 = tf.Variable(zeros_initializer([32]))

        # Define first layer outputs
        y1_ = tf.matmul(x, W1) + b1
        y1 = tf.nn.relu(y1_)

        # Initialise second layer weights
        W2 = tf.Variable(initializer([32, 32]))
        b2 = tf.Variable(zeros_initializer([32]))

        # Define second layer outputs
        y2_ = tf.matmul(y1, W2) + b2
        y2 = tf.nn.relu(y2_)

        # Initialise third layer weights
        W3 = tf.Variable(initializer([32, 10]))
        b3 = tf.Variable(zeros_initializer([10]))

        # Define last layer linear output
        y = tf.matmul(y2, W3) + b3

        # Define the cross-entropy loss after applying the softmax
        loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        )

        # Define the optimizer
        train_object = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Accuracy evaluation
        # Get 20% random images for the train accuracy
        image_num = int(0.2 * eval_mnist.train.labels.shape[0])
        indices = np.random.choice(eval_mnist.train.labels.shape[0], image_num)

        # Get a vector of which values are correctly classified
        correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        # Get the accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        # Train.
        i, train_accuracy, test_accuracy = 0, [], []
        with tf.train.MonitoredSession() as sess:
            while mnist.train.epochs_completed < num_epochs:

                # Update.
                i += 1
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)

                #################
                # Training step #
                #################
                sess.run(train_object, feed_dict={x: batch_xs, y_: batch_ys})

                #####################################


if __name__=="__main__":
    main_func()