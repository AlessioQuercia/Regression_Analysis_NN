from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


######## READ DATASET ########

def read_dataset(filepath, sep, target):
    data = pd.read_csv(filepath, sep=sep).to_dict()
    for k, v in data.items():
        temp = []
        for a, b in v.items():
            temp.append(b)
        data[k] = np.array(temp)
    labels = np.array(data.pop(target))

    for k, v in data.items():
        if ' ' in k:
            data[k.replace(' ', '_')] = data.pop(k)
    return data, labels

filepath = "C:/Users/Alessio/Desktop/winequality-red.csv"
data, labels = read_dataset(filepath, ';', "quality")

train_x = data
train_y = labels


######## WRITE AN INPUT FUNCTION ########

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # next_item = dataset.make_one_shot_iterator().get_next()
    # sess = tf.Session()
    # with sess.as_default():
    #     while True:
    #         try:
    #             dictio = next_item[0]
    #             for k, v in dictio.items():
    #                 print(sess.run(tf.shape(v)))
    #             print(sess.run(next_item))
    #         except tf.errors.OutOfRangeError:
    #             break
    #         break

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


######## CREATE FEATURE COLUMNS ########

# You must define your model's feature columns to specify how the model should use each feature.
# Whether working with pre-made Estimators or custom Estimators, you define feature columns in the same fashion.
# The following code creates a simple numeric_column for each input feature, indicating that the value of the
# input feature should be used directly as an input to the model:
# Feature columns describe how to use the input.

def get_normalization_parameters(traindf):
    # traindf is the dictionary containing the data
    # features is the list of the features' names
    """Get the normalization parameters (E.g., mean, std) for traindf for
    features. We will use these parameters for training, eval, and serving."""

    def _z_score_params(column):
        mean = traindf[column].mean()
        std = traindf[column].std()
        return {'mean': mean, 'std': std}

    normalization_parameters = {}
    for column, v in traindf.items():
        normalization_parameters[column] = _z_score_params(column)
    return normalization_parameters

def _make_zscaler(mean, std):
    def zscaler(col):
        return (col - mean) / std

    return zscaler

def create_feature_columns(train_x, use_normalization):
    my_feature_columns = []
    normalizer_fn = None
    for key in train_x.keys():
        if use_normalization:
            column_params = normalization_parameters[key]
            mean = column_params['mean']
            std = column_params['std']
            normalizer_fn = _make_zscaler(mean, std)
        my_feature_columns.append(tf.feature_column.numeric_column(key=key, normalizer_fn=normalizer_fn))
    return my_feature_columns

normalization_parameters = get_normalization_parameters(train_x)

my_feature_columns = create_feature_columns(train_x, True)


######## WRITE A MODEL FUNCTION ########

def my_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration

    ######## DEFINE THE MODEL ########

    # DEFINE THE INPUT LAYER
    # Use `input_layer` to apply the feature columns.
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    # DEFINE THE INPUT LAYER
    # Build the hidden layers, sized according to the 'hidden_units' param.
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # DEFINE THE OUTPUT LAYER
    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # DEFINE THE PREDICT MODE
    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # CALCULATE THE LOSS
    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # DEFINE THE EVALUATE MODE
    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # DEFINE THE TRAIN MODE
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


######## THE CUSTOM ESTIMATOR ########
# Build 2 hidden layer DNN with 10, 10 units respectively.
classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    params={
        'feature_columns': my_feature_columns,
        # Two hidden layers of 10 nodes each.
        'hidden_units': [10, 10],
        # The model must choose between 3 classes.
        'n_classes': 10,
    },
    model_dir="./model_dir")


######## TRAIN THE MODEL ########

# print(train_x)
# print(train_y)

batch_size = 1
train_steps = 500

history = classifier.train(
    input_fn=lambda: train_input_fn(train_x, train_y, batch_size),
    steps=train_steps)

evaluation = classifier.evaluate(
    input_fn=lambda: train_input_fn(train_x, train_y, batch_size),
    steps=train_steps)

print(evaluation)

