from __future__ import print_function
import tensorflow as tf
import json
import numpy as np

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
num_input = 0
num_output = 0

# tf Graph input
def get_X_Y(num_input, num_output):
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_output])
    return X, Y

# Store layers weight & bias
def get_weights(num_input, n_hidden_1, n_hidden_2, num_output):
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, num_output]))
    }
    return weights

def get_biases(n_hidden_1, n_hidden_2, num_output):
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_output]))
    }
    return biases

# Create model
def neural_net(x):
    weights = get_weights(num_input, n_hidden_1, n_hidden_2, num_output)
    biases = get_biases(n_hidden_1, n_hidden_2, num_output)
    # Hidden fully connected layer with n_hidden_1 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with n_hidden_2 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
def get_predictions(X):
    preds = neural_net(X)
    return preds

# Evaluate model
def get_loss(X, Y):
    preds = get_predictions(X)
    loss_op = tf.reduce_mean(tf.losses.mean_squared_error(
        predictions=preds, labels=Y))
    return loss_op

def get_accuracy(X, Y):
    preds = get_predictions(X)
    correct_pred = tf.equal(preds, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

# Define loss and optimizer
def get_train_op(X, Y, learning_rate):
    loss_op = get_loss(X, Y)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    return train_op

def run_model(input_train_data, output_train_data, input_test_data, output_test_data):
    # Get num_input & num_output to construct model
    global num_input
    num_input = int(num_frames_to_input * input_train_data.shape[1])
    global num_output
    num_output = int(output_train_data.shape[1])
    print("NN input & output dimension")
    print(num_input, num_output)

    # Obtain placeholders for inputs and outputs
    X, Y = get_X_Y(num_input, num_output)

    # Obtain different operations
    train_op = get_train_op(X, Y, learning_rate)
    loss_op = get_loss(X, Y)
    accuracy = get_accuracy(X, Y)
    get_preds = get_predictions(X)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Write summaries for tensorboard visualization
        writer = tf.summary.FileWriter('../logs', graph=tf.get_default_graph())

        for step in range(1, num_steps + 1):
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            indices = np.random.randint(0, input_train_data.shape[0], size=batch_size)
            batch_x = input_train_data[indices, :]
            batch_y = output_train_data[indices, :]
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
                preds = sess.run(get_preds, feed_dict={X: batch_x})
            if step % 5000 == 0:
                print("Preds:", preds[0, 0:5])
                print("Truth:", batch_y[0, 0:5])

        print("Optimization Finished!")

        # Calculate accuracy for MNIST test images
        print("Testing Accuracy:", \
              sess.run(accuracy, feed_dict={X: input_test_data,
                                            Y: output_test_data}))