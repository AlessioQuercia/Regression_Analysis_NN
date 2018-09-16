import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_dataset(filePath, delimiter=','):
    return np.genfromtxt(filePath, delimiter=delimiter)


def build_model(train_data):
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.GradientDescentOptimizer(0.001)

    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae']
    )
    return model

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

# filepath = "C:/Users/Alessio/Desktop/wdbc.data"
# dataset = read_dataset(filepath)
# print(dataset)

# Wine Quality Dataset
# filepath = "C:/Users/Alessio/Desktop/winequality-red.csv"
# dataset = pd.read_csv(filepath, sep=';')
# dataset = dataset.values

# Boston Housing Dataset
filepath = "C:/Users/Alessio/Desktop/housing.data"
dataset = pd.read_csv(filepath, delim_whitespace=True,
           skipinitialspace=True)
dataset = dataset.values
print(dataset)

print(dataset.shape)

data = np.delete(dataset, 11, axis=1)
labels = np.array([row[13] for row in dataset])

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

cut = int(round(data.shape[0]*0.7))

training_X = np.array(data[:cut])
training_Y = np.array(labels[:cut])


# print(training_X.shape)
# print(training_Y.shape)

test_X = np.array(data[cut:])
test_Y = np.array(labels[cut:])

# print(test_X.shape)
# print(test_Y.shape)

# Normalization
mean = training_X.mean(axis=0)
std = training_X.std(axis=0)

training_X = (training_X - mean) / std
# training_Y = training_Y/10
test_X = (test_X - mean) / std
# test_Y = test_Y/10

order2 = np.argsort(np.random.random(training_Y.shape))

training_X = training_X[order2]
training_Y = training_Y[order2]

cut2 = int(round(training_X.shape[0]*0.8))

trainMinusVal_X = training_X[:cut2]
trainMinusVal_Y = training_Y[:cut2]

validation_X = training_X[cut2:]
validation_Y = training_Y[cut2:]

# print(trainMinusVal_Y.shape)
# print(validation_Y.shape)

model = build_model(training_X)
model.summary()

EPOCHS = 500

# The patience parameter is the amount of epochs to check for improvement.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(training_X, training_Y, epochs=EPOCHS, verbose=0, validation_split=0.2, callbacks=[early_stop])

plot_history(history)

[loss, mae] = model.evaluate(test_X, test_Y, verbose=0)

print("Testing set Mean Abs Error: {:7.2f}".format(mae))

test_predictions = model.predict(test_X).flatten()

print(test_predictions[:10])

print(test_Y[:10])