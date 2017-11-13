from __future__ import print_function
import tensorflow as tf
import json
import numpy as np

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

settings_file = '../settings/model_settings.json'
settings = json.load(open(settings_file))

# Parameters
learning_rate = settings['init_learning_rate']
batch_size = settings['batch_size']
num_steps = settings['num_steps']
display_step = settings['display_step']

# Network Parameters
num_frames_to_input = settings['num_frames_to_input']
n_hidden_1 = settings['n_hidden_1']
n_hidden_2 = settings['n_hidden_2']


# tf Graph input
def get_X_Y(num_input, num_output):
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_output])
    return X, Y


# Create model
def neural_net(x, weights, biases):
    # Hidden fully connected layer with n_hidden_1 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with n_hidden_2 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
def get_predictions(X, weights, biases):
    preds = neural_net(X, weights, biases)
    return preds


# Evaluate model
def get_loss(X, weights, biases, Y):
    preds = get_predictions(X, weights, biases)
    loss_op = tf.reduce_min(tf.losses.mean_squared_error(
        predictions=preds, labels=Y))
    return loss_op


def get_accuracy(X, weights, biases, Y):
    preds = get_predictions(X, weights, biases)
    correct_pred = tf.equal(preds, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


# Define loss and optimizer
def get_train_op(X, weights, biases, Y, learning_rate):
    loss_op = get_loss(X, weights, biases, Y)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    return train_op


def run_model(input_train_data, output_train_data, input_test_data, output_test_data):
    # def run_model():
    # Get num_input & num_output to construct model
    num_input = int(num_frames_to_input * input_train_data.shape[1])
    num_output = int(output_train_data.shape[1])
    print("NN input & output dimension")
    print(num_input, num_output)

    ## Initilize weights & biases ##
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, num_output]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_output]))
    }

    # Obtain placeholders for inputs and outputs
    X, Y = get_X_Y(num_input, num_output)

    # Get predictions
    preds = neural_net(X, weights, biases)

    # Get different operations
    loss_op = tf.reduce_min(tf.losses.mean_squared_error(
        predictions=preds, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # # Obtain different operations
    # train_op = get_train_op(X, weights, biases, Y, learning_rate)
    # loss_op = get_loss(X, weights, biases, Y)
    # accuracy = get_accuracy(X, weights, biases, Y)
    # get_preds = get_predictions(X, weights, biases)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.device('/device:GPU:0'):

            # Run the initializer
            sess.run(init)

            # Write summaries for tensorboard visualization
            # writer = tf.summary.FileWriter('../logs', graph=tf.get_default_graph())

            for step in range(1, num_steps + 1):
                # batch_x, batch_y = mnist.train.next_batch(batch_size)
                indices = np.random.randint(0, input_train_data.shape[0], size=batch_size)
                batch_x = input_train_data[indices, :]
                batch_y = output_train_data[indices, :]
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss))
                    predicts = sess.run(preds, feed_dict={X: batch_x})
                    W = sess.run(weights)
                    print("first wt: ", W['h1'][0, 0])
                if step % (50 * display_step) == 0:
                    print("Preds:", predicts[0, 0:5])
                    print("Truth:", batch_y[0, 0:5])

            print("Optimization Finished!")

            # Calculate accuracy for MNIST test images
            # print("Testing Accuracy:", \
            #       sess.run(accuracy, feed_dict={X: input_test_data,
            #                                     Y: output_test_data}))
            # print("Testing Accuracy:", \
            #       sess.run(accuracy, feed_dict={X: mnist.test.images,
            #                                     Y: mnist.test.labels}))
