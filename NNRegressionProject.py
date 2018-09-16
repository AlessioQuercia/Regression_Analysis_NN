from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.preprocessing import MinMaxScaler


######## READ DATASET ########
def read_dataset_as_matrix(filepath, sep, target):
    data = pd.read_csv(filepath, sep=sep)
    # print(data.info())
    labels = data.pop(target).values    # == as_matrix()
    data = data.values                  # == as_matrix()
    return data, labels


######## NORMALIZE MATRIX ########
def normalize(data):
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(data)
    return norm_data


######## DENORMALIZE MATRIX ########
def denormalize(orig_data, norm_data):
    scaler = MinMaxScaler()
    orig_data = orig_data.reshape(-1, 1)
    norm_data = norm_data.reshape(-1, 1)
    scaler.fit_transform(orig_data)
    data = scaler.inverse_transform(norm_data)
    return data


######## NEURAL NETWORK MODEL ########
def neural_net_model(x_data, input_dim, hidden_dim, output_dim, hidden_layers_num, activation_fn):
    # DEFINING INPUT LAYER
    input_weights = tf.Variable(tf.random_uniform([input_dim, hidden_dim]))
    input_bias = tf.Variable(tf.random_uniform([hidden_dim]))
    input_layer = tf.add(tf.matmul(x_data, input_weights), input_bias)
    input_layer = activation_fn(input_layer)
    print("inputs")
    print("hidden_layer_0 = input_layer (weights between inputs and first hidden layer)")

    # DEFINING HIDDEN LAYERS
    hidden_layers_num = max(0, hidden_layers_num - 1)   # At least 1 hidden layer, the input_layer, is created
    hidden_weights = []
    hidden_biases = []
    hidden_layers = []
    hidden_layers.append(input_layer)

    for i in range(hidden_layers_num):
        print("hidden_layer_" + str(i + 1))
        hidden_weights.append(tf.Variable(tf.random_uniform([hidden_dim, hidden_dim])))
        hidden_biases.append(tf.Variable(tf.random_uniform([hidden_dim])))
        hidden_layers.append(activation_fn(tf.add(tf.matmul(hidden_layers[i], hidden_weights[i]), hidden_biases[i])))
        # if i == 0:
        #     hidden_weights.append(tf.Variable(tf.random_uniform([hidden_dim, hidden_dim])))
        #     hidden_biases.append(tf.Variable(tf.random_uniform([hidden_dim])))
        #     hidden_layers.append(activation_fn(tf.add(tf.matmul(input_layer, hidden_weights[i]), hidden_biases[i])))
        # else:
        #     hidden_weights.append(tf.Variable(tf.random_uniform([hidden_dim, hidden_dim])))
        #     hidden_biases.append(tf.Variable(tf.random_uniform([hidden_dim])))
        #     hidden_layers.append(activation_fn(tf.add(tf.matmul(hidden_layers[i - 1], hidden_weights[i]), hidden_biases[i])))

    # DEFINING OUTPUT LAYER
    # if (hidden_layers_num > 0): # At least 2 hidden layers
    output_weights = tf.Variable(tf.random_uniform([hidden_dim, output_dim]))
    output_biases = tf.Variable(tf.random_uniform([output_dim]))
    output = tf.add(tf.matmul(hidden_layers[-1], output_weights), output_biases)
    # else: # Only 1 hidden layer (the one connected to the input and then to the output)
    #     output_weights = tf.Variable(tf.random_uniform([hidden_dim, output_dim]))
    #     output_biases = tf.Variable(tf.random_uniform([output_dim]))
    #     output = tf.add(tf.matmul(input_layer, output_weights), output_biases)
    print("output_layer (weights between last hidden layer and output layer)")
    print("outputs")

    # FOR LOGISTIC REGRESSION
    # output = tf.sigmoid(output)

    return output



#### EXTRACTING DATASET

# Wine Quality Dataset
filepath = "C:/Users/Alessio/Desktop/winequality-red.csv"
data, labels = read_dataset_as_matrix(filepath, ';', "quality")

# # Boston Housing Dataset
# filepath = "C:/Users/Alessio/Desktop/housing.data"
# data = pd.read_csv(filepath, delim_whitespace=True,
#            skipinitialspace=True)
# labels = data.pop('MEDV').values
# data = data.values

# # Wisconsin Diagnostic Breast Cancer Data Set
# filepath = "C:/Users/Alessio/Desktop/wdbc.data"
# data = pd.read_csv(filepath, sep=",")
# data.pop('ID_number')
# labels = data.pop('Diagnosis').values
# data = data.values
# for i in range(labels.shape[0]):
#     if labels[i] == 'M':
#         labels[i] = 0.0
#     else:
#         labels[i] = 1.0

# print(data)
# print(labels)


# # Shuffle the data and the labels
# order = np.argsort(np.random.random(labels.shape))
#
# data = data[order]
# labels = labels[order]


#### PARAMETERS INITIALIZATION
validation_epochs = 100
training_epochs = 1000

training_errors = []
test_errors = []
maes = []
accuracies = []

data_dim = data.shape[0]
train_dim = int(data.shape[0] * 0.8)    # 80% del data set
test_dim = data_dim - train_dim         # 20% del data set

start = train_dim + 1
end = data_dim


#### DEFINING COST FUNCTION

xs = tf.placeholder("float")
ys = tf.placeholder("float")
learning_rate = tf.placeholder(tf.float32, shape=[])

input_nodes_number = 11
output_nodes_number = 1
# hidden_nodes_number = int((input_nodes_number + output_nodes_number)/2)
# hidden_layers_number = 1
hidden_nodes_number = 2
hidden_layers_number = 3

activation_fn = tf.nn.softplus

output = neural_net_model(xs, input_nodes_number, hidden_nodes_number, output_nodes_number,
                          hidden_layers_number, activation_fn)     # Output del modello su input xs

mse_loss = tf.reduce_mean(tf.square(output - ys))           # Mean Square Error (MSE)

mae_loss = tf.reduce_mean(tf.abs(output - ys))              # Mean Absolute Error (MAE)

# num = tf.log(1 + tf.exp(-(output*ys)))
#
# den = tf.log(tf.constant(2, dtype=num.dtype))
#
# log_loss = tf.reduce_sum(num/den)                           # Logaritmic Loss (-1, +1)

# FOR LOGISTIC REGRESSION
log_loss = - tf.reduce_mean(ys * tf.log(output) + (1 - ys) * tf.log(1 - output))        # Logaritmic Loss

loss = mae_loss

# loss = tf.losses.absolute_difference(output, ys)          # Absolute Difference

# loss = tf.losses.huber_loss(output, ys)                   # Huber loss (= Smooth Mean Absolute Error)

# loss = tf.losses.log_loss(output, ys)                     # Log loss

mae, mae_update = tf.metrics.mean_absolute_error(ys, output, name="mae_metric")           # Mean Absolute Error (MAE)

mae_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="mae_metric")  # Mae local variables

mae_vars_initializer = tf.variables_initializer(var_list=mae_vars)

train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)      # Gradient descent step


#### TEST LOOP
while start >= 0:
    test_sub_arr = []
    for a in range(start, end):
        test_sub_arr.append(a)

    train_x = np.delete(data, test_sub_arr, axis=0)  # 80% del data set
    train_y = np.delete(labels, test_sub_arr)        # 80% del label set

    test_x = data[start:end]                         # 20% del data set
    test_y = labels[start:end]                       # 20% del label set

    max_y = 100

    #### NORMALIZATION
    train_x = normalize(train_x)
    # train_y = normalize(train_y.reshape(-1, 1))
    # train_x = (train_x - train_x.mean())/train_x.std()
    # train_y = train_y/max_y
    # orig_test_y = test_y
    test_x = normalize(test_x)
    # test_y = normalize(test_y.reshape(-1, 1))
    # test_x = (test_x - test_x.mean())/test_x.std()
    # test_y = test_y/max_y

    # print(train_x, train_y)

    # print("train: ", train_x[0], train_y)
    # print("test: ", test_x[0], test_y)

    # print(start, end)

    start -= test_dim
    end -= test_dim

    #### VALIDATION PARAMETERS INITIALIZATION


    # # Shuffle the training
    # order2 = np.argsort(np.random.random(train_y.shape))
    #
    # train_x = train_x[order2]
    # train_y = train_y[order2]

    train_minus_val_dim = int(train_x.shape[0] * 0.8)  # 80% del training set
    val_dim = train_x.shape[0] - train_minus_val_dim  # 20% del training set

    val_sub_arr = []
    for b in range(train_minus_val_dim, train_x.shape[0]):
        val_sub_arr.append(b)

    train_minus_val_x = np.delete(train_x, val_sub_arr, axis=0)     # 80% dei dati del training set
    train_minus_val_y = np.delete(train_y, val_sub_arr)             # 80% delle label del training set

    val_x = train_x[train_minus_val_dim:train_x.shape[0]]           # 20% delle label del training set
    val_y = train_y[train_minus_val_dim:train_x.shape[0]]           # 20% delle label del training set

    # print("Train_minus_val: ", train_minus_val_x[0], train_minus_val_y)
    # print("Val: ", val_x[0], val_y)

    validation_errors = []
    learning_rates = []

    learn_rate = 0.0001

    validation_tolerance = 0.3
    #### VALIDATION LOOP (find a good learning rate value)
    for j in range(validation_epochs):

        learning_rates.append(learn_rate)

        #### Train the model on the train_x_minus_val_x set and test it on val_x

        # val_train = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

        #### VALIDATION SESSION
        with tf.Session() as sess:
            # Initiate session and initialize all variables
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            # Train the model using each sample (train_minus_val_x[s], train_minus_val_y[s]) of the train_minus_val set
            # BATCH TRAINING (UPDATE AFTER COMPUTING AN ENTIRE BATCH, BATCH_SIZE = TRAIN_MINUS_VAL_X_SIZE)
            # sess.run(train_step, feed_dict={xs: [train_minus_val_x[s]], ys: train_minus_val_y[s], learning_rate: learn_rate})
            # ONLINE TRAINING (UPDATE EACH SAMPLE)
            for s in range(train_minus_val_x.shape[0]):
                sess.run(train_step, feed_dict={xs: [train_minus_val_x[s]], ys: train_minus_val_y[s], learning_rate: learn_rate})

            # Test the trained model on the validation test:
            validation_errors.append(sess.run(loss, feed_dict={xs: val_x, ys: val_y}))      # VALIDATION ERROR

        #### Update the learning_rate for the next step
        learn_rate += 0.0001

    min_validation_cost = min(validation_errors)
    validated_learning_rate = learning_rates[validation_errors.index(min_validation_cost)]

    # test_train = tf.train.GradientDescentOptimizer(learning_rate=validated_learning_rate).minimize(cost)

    done = False

    while not done:

        #### TRAINING - TEST SESSION
        with tf.Session() as sess:
            # Initiate session and initialize all variables
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            training_tolerance = 0.05
            training_patience = 5

            # # FOR LOGISTIC REGRESSION
            # current_min_loss_train = np.inf

            current_min_loss_train = np.inf
            current_min_mae_train = np.inf
            last_min_update_epoch = -1

            # TRAIN THE MODEL
            for z in range(training_epochs):
                sess.run(mae_vars_initializer)

                # Train the model on the train_x set (using the validated_learning_rate just obtained) and test it on test_x
                # BATCH TRAINING (UPDATE AFTER COMPUTING AN ENTIRE BATCH, BATCH_SIZE = TRAIN_X_SIZE)
                # sess.run(train_step, feed_dict={xs: train_x, ys: train_y, learning_rate: validated_learning_rate})
                # ONLINE TRAINING (UPDATE EACH SAMPLE)
                for smp in range(train_x.shape[0]):
                    sess.run(train_step, feed_dict={xs: [train_x[smp]], ys: train_y[smp], learning_rate: validated_learning_rate})

                # Compute the training error on the training set:
                [mae_train_update, loss_train] = sess.run([mae_update, loss], feed_dict={xs: train_x, ys: train_y})    # TRAINING ERROR

                if np.isnan(loss_train):
                    print(loss_train)
                    break


                # stream_vars = [i for i in tf.local_variables()]
                # print('[total, count]:', sess.run(stream_vars))
                # print(train_x.shape[0])

                # Compute the the Mean Absolute Training Error
                mae_train = sess.run(mae)

                print("Epoch: " + str(z + 1) + "    Mean Absolute Training Error: " + str(
                    mae_train) + "    Mean Squared Training Error: " + str(loss_train))
                training_errors.append(loss_train)

                # # FOR LOGISTIC REGRESSION
                # if loss_train < current_min_loss_train:
                # If the loss_train just computed is inferior to the current_min_loss_train, update this last one
                if loss_train < current_min_loss_train:
                    current_min_loss_train = loss_train
                    current_min_mae_train = mae_train
                    last_min_update_epoch = z

                # The training error hasn't changed for training_patience epochs, so we break the training loop
                if z - last_min_update_epoch > training_patience or mae_train <= training_tolerance:
                    break

            if np.isnan(loss_train):
                continue


            print("Mean Absolute Training Error: " + str(current_min_mae_train) +
                  "    Mean Squared Test Error: " + str(current_min_loss_train))

            sess.run(mae_vars_initializer)

            # TEST THE TRAINED MODEL
            # Compute the test error on the test set:
            [mae_test_update, loss_test] = sess.run([mae_update, loss], feed_dict={xs: test_x, ys: test_y})   # TEST ERROR

            mae_test = sess.run(mae)

            print("Mean Absolute Test Error: " + str(mae_test) + "    Mean Squared Test Error: " + str(loss_test))
            test_errors.append(loss_test)
            maes.append(mae_test)

            # PREDICT THE TEST SET'S LABELS USING THE TRAINED MODEL
            pred = sess.run(output, feed_dict={xs: test_x})
            # denormed_pred = denormalize(orig_test_y, pred)
            # denormed_test_y = denormalize(orig_test_y, test_y)

            # FOR LOGISTIC REGRESSION
            # correct = 0
            # for m in range(test_x.shape[0]):
            #     if pred[m] > 0.5:
            #         pred[m] = 1.0
            #     else:
            #         pred[m] = 0.0
            #
            #     if pred[m] == test_y[m]:
            #         correct += 1
            #     else:
            #         print(pred[m], test_y[m])
            #
            #
            # # accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, test_y), dtype=tf.float32))
            #
            # # acc = sess.run(accuracy)
            #
            # acc = correct/test_x.shape[0]
            #
            # accuracies.append(acc)

            done = True


        for m in range(test_x.shape[0]):
            print(pred[m], test_y[m])


    # #### PLOT THE PREDICTIONS
    #
    # #### SAVE THE MODEL
    # if input('Save model ? [Y/N]') == 'Y':
    #     name = "./models/" + input() + '.ckpt'
    #     saver.save(sess, name)
    #     print('Model Saved')


#### Compute accuracy on the results given by the different models on the different test_set

sum = 0
for test_er in test_errors:
    print(test_er)
    sum += test_er
mmse = sum/len(test_errors)
print("Mean Mean Squared Error: ", mmse)    # Mean Test Error (each computed as Mean Square Error)

sum = 0
for er in maes:
    print(er)
    sum += er
mmae = sum/len(maes)
print("Mean Mean Absolute Error: ", mmae)    # Mean Test Error (each computed as Mean Absolute Error)

# FOR LOGISTIC REGRESSION
# sum = 0
# for a in accuracies:
#     print(a)
#     sum += a
# macc = sum/len(accuracies)
# print("Mean Accuracy: ", macc)