from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
            label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
    plt.legend()
    plt.ylim([0,5])
    plt.tight_layout()
    plt.savefig('simple_example')
    plt.show()

def read_from_csv(filepath):
    dataset = pd.read_csv(filepath, sep=';')
    dataset = dataset.values

    # print(dataset)
    # print(dataset.shape)

    data = np.delete(dataset, 11, axis=1)
    labels = np.array([row[11] for row in dataset])

    # print(data)
    # print(labels)
    #
    # print(dataset.shape)
    # print(data.shape)
    # print(labels.shape)

    # Shuffle the sets
    order = np.argsort(np.random.random(labels.shape))
    data = data[order]
    labels = labels[order]

    cut = int(round(data.shape[0] * 0.7))

    training_X = np.array(data[:cut])
    training_Y = np.array(labels[:cut])

    # print(training_X.shape)
    # print(training_Y.shape)

    test_X = np.array(data[cut:])
    test_Y = np.array(labels[cut:])

    # print(test_X.shape)
    # print(test_Y.shape)

    mean = training_X.mean(axis=0)
    std = training_X.std(axis=0)

    training_X = (training_X - mean) / std
    test_X = (test_X - mean) / std

    order2 = np.argsort(np.random.random(training_Y.shape))

    training_X = training_X[order2]
    training_Y = training_Y[order2]

    cut2 = int(round(training_X.shape[0] * 0.8))

    trainMinusVal_X = training_X[:cut2]
    trainMinusVal_Y = training_Y[:cut2]

    validation_X = training_X[cut2:]
    validation_Y = training_Y[cut2:]

    # print(trainMinusVal_Y.shape)
    # print(validation_Y.shape)

    return training_X, training_Y, test_X, test_Y, validation_X, validation_Y, trainMinusVal_X, trainMinusVal_Y


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
    for column, v  in traindf.items():
        normalization_parameters[column] = _z_score_params(column)
    return normalization_parameters

def _make_zscaler(mean, std):
    def zscaler(col):
        return (col - mean) / std

    return zscaler

def create_feature_columns(train_x, use_normalization):
    my_feature_columns = []
    for key in train_x.keys():
        normalizer_fn = None
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

    # DEFINE THE HIDDEN LAYERS
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
    # prediction = tf.nn.softmax(logits[0]) * 10
    # loss = tf.losses.mean_squared_error(labels=labels, predictions=prediction)

    # DEFINE THE EVALUATE MODE
    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # DEFINE THE TRAIN MODE
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


######## CREATE A CUSTOM ESTIMATOR ########

# Build 2 hidden layer DNN with 10, 10 units respectively.
classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    params={
        'feature_columns': my_feature_columns,
        # Two hidden layers of 10 nodes each.
        'hidden_units': [10, 10],
        # The model must choose between 3 classes.
        'n_classes': 1,
    },
    model_dir="./model_dir")


######## TRAIN THE MODEL ########

# print(train_x)
# print(train_y)

# batch_size = 1
# train_steps = 500
#
# history = classifier.train(
#     input_fn=lambda: train_input_fn(train_x, train_y, batch_size),
#     steps=train_steps)
#
# evaluation = classifier.evaluate(
#     input_fn=lambda: train_input_fn(train_x, train_y, batch_size),
#     steps=train_steps)
#
# print(evaluation)


def neural_net_model(x_data, input_dim, hidden_dim, output_dim, hidden_layers_num, activation_fn):
    # DEFINING INPUT LAYER
    input_weights = tf.Variable(tf.random_uniform([input_dim, hidden_dim]))
    input_bias = tf.Variable(tf.zeros([hidden_dim]))
    input_layer = tf.add(tf.matmul(x_data, input_weights), input_bias)
    input_layer = activation_fn(input_layer)

    # DEFINING HIDDEN LAYERS
    hidden_layers_num = max(1, hidden_layers_num)
    hidden_weights = []
    hidden_biases = []
    hidden_layers = []

    for i in range(1, hidden_layers_num):
        # print("hidden_layer_", i)
        if i == 1:
            hidden_weights.append(tf.Variable(tf.random_uniform([hidden_dim, hidden_dim])))
            hidden_biases.append(tf.Variable(tf.random_uniform([hidden_dim])))
            hidden_layers.append(activation_fn(tf.add(tf.matmul(input_layer, hidden_weights[0]), hidden_biases[0])))
        else:
            hidden_weights.append(tf.Variable(tf.random_uniform([hidden_dim, hidden_dim])))
            hidden_biases.append(tf.Variable(tf.random_uniform([hidden_dim])))
            hidden_layers.append(activation_fn(tf.add(tf.matmul(hidden_layers[i - 1 - 1], hidden_weights[i - 1]), hidden_biases[i - 1])))

    # DEFINING OUTPUT LAYER
    output_weights = tf.Variable(tf.random_uniform([hidden_dim, output_dim]))
    output_biases = tf.Variable(tf.random_uniform([output_dim]))
    output = tf.add(tf.matmul(hidden_layers[i - 1 - 1], output_weights), output_biases)

    return output

xs = tf.placeholder("float")
ys = tf.placeholder("float")

# Normalization
for k, v in train_x.items():
    v_arr = np.array(v)
    normalized_v = (v_arr - v_arr.mean())/v_arr.std()
    train_x[k] = normalized_v

max_y = 10
train_y = train_y/max_y;

data_shape = 0
for k, v in train_x.items():
    data_shape = len(v)
    print(data_shape)
    break

train_x_shape = int(data_shape * 0.7)
test_x_shape = data_shape - train_x_shape

learning_rate = 0.0001
training_epochs = 1000

test_x = train_x
test_y = train_y

output = neural_net_model(xs, 11, 5, 1, 10, tf.nn.relu)

print(output, ys)

cost = tf.reduce_mean(tf.square(output-ys))

# our mean squared error cost function

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Gradinent Descent optimiztion just discussed above for updating weights and biases

with tf.Session() as sess:
    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sess,'yahoo_dataset.ckpt')

    c_t = []
    c_test = []
    for i in range(training_epochs):
        for j in range(train_x_shape):
            x_sample = []
            for k, v in train_x.items():
                x_sample.append(v[j])
            sess.run([cost, train], feed_dict={xs:[x_sample], ys:train_y[j]})
            # print(sess.run(output, feed_dict={xs:[x_sample], ys:train_y[j]}), train_y[j])

            # Run cost and train with each sample

        x_samples_train = []
        for u in range(train_x_shape):
            x_samples_train.append([])
        for k, v in train_x.items():
            for u in range(train_x_shape):
                x_samples_train[u].append(v[u])

        c_t.append(sess.run(cost, feed_dict={xs:x_samples_train, ys:train_y[:train_x_shape]}))

        x_samples_test = []
        for u in range(test_x_shape):
            x_samples_test.append([])
        for k, v in test_x.items():
            for u in range(test_x_shape):
                x_samples_test[u].append(v[u + train_x_shape])

        c_test.append(sess.run(cost, feed_dict={xs:x_samples_test, ys:test_y[train_x_shape:]}))
        print('Epoch :', i, 'Cost_train :', c_t[i], 'Cost_test :', c_test[i])

    pred = sess.run(output, feed_dict={xs:x_samples_test})
    # predict output of test data after training

    for l in range(len(test_y[train_x_shape:])):
        print(pred[l], test_y[l])

    print('Cost :', sess.run(cost, feed_dict={xs:x_samples_test, ys:test_y[test_x_shape:]}))