from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
#
# print(type(mnist.train))

settings_file = '../settings/model_settings.json'
settings = json.load(open(settings_file))

# Parameters
learning_rate = settings['init_learning_rate']
batch_size = settings['batch_size']
num_steps = settings['num_steps']

# Network Parameters
num_frames_to_input = 9
n_hidden_1 = settings['n_hidden_1']
n_hidden_2 = settings['n_hidden_2']
num_input = num_frames_to_input*settings['fft_size']
num_output = settings['fft_size']

# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['ds_frames']
    # Hidden fully connected layer 1
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer 2
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_output)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    preds = neural_net(features)

    # Predictions
    # pred_classes = tf.argmax(logits, axis=1)
    # pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    # loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    loss_op = tf.reduce_mean(tf.losses.mean_squared_error(
        predictions=preds, labels=tf.cast(labels, dtype=float)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=preds)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=preds,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

def train_fn(train_data_input_frames, train_data_output_frames):
    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'ds_frames': train_data_input_frames}, y=train_data_output_frames,
        batch_size=batch_size, num_epochs=None, shuffle=True)
    model = tf.estimator.Estimator(model_fn)
    model.train(input_fn, steps=num_steps)

def evaluate_fn(test_data_input_frames, test_data_output_frames):
    # Define the input function for evaluating
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'ds_frames': test_data_input_frames}, y=test_data_output_frames,
        batch_size=batch_size, shuffle=False)
    # Use the Estimator 'evaluate' method
    model = tf.estimator.Estimator(model_fn)
    model.evaluate(input_fn)