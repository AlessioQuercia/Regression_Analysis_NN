from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


########## GRAPH ##########
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)


########## TENSORBOARD ##########
# writer = tf.summary.FileWriter('.', tf.get_default_graph())
# writer.add_graph(tf.get_default_graph())


########## SESSION ##########
sess = tf.Session()
print(sess.run(total))
print(sess.run({'ab': (a, b), 'total': total}))

vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))


########## FEEDING ##########
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

# writer = tf.summary.FileWriter('.', tf.get_default_graph())

print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))


########## DATASETS ##########
my_data = [
    [0, 1,],
    [2, 3,],
    [4, 5,],
    [6, 7,],
]

slices = tf.data.Dataset.from_tensor_slices(my_data)

next_item = slices.make_one_shot_iterator().get_next()

while True:
  try:
      print(sess.run(next_item))
  except tf.errors.OutOfRangeError:
      break

#same as before, but initializing the iterator and random dataset from Normal distribution
r = tf.random_normal([10, 3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)
while True:
  try:
    print(sess.run(next_row))
  except tf.errors.OutOfRangeError:
    break


########## LAYERS ##########
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)


########## INITIALIZING LAYERS ##########
init = tf.global_variables_initializer()
sess.run(init)


########## EXEVUTING LAYERS ##########
print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}))


########## LAYER FUNCTION SHORTCUTS ##########
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.layers.dense(x, units=1)

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))


########## FEATURE COLUMNS ##########
features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)

var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))

print(sess.run(inputs))


########## TRAINING ##########

#Define the data
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

#Define the model
linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

#Evaluate predictions (without training)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y_pred))

#Loss
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
print(sess.run(loss))

#Training
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)


########## TENSORS ##########

#### RANK ####

#Rank 0
mammal = tf.Variable("Elephant", tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

#Rank 1
my_str = tf.Variable(["Hello"], tf.string)
cool_numbers = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)

#Higher ranks
my_mat = tf.Variable([[7], [11]], tf.int16)
my_xor = tf.Variable([[False, True], [True, False]], tf.bool)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
my_matC = tf.Variable([[7], [11]], tf.int32)
my_image = tf.zeros([10, 299, 299, 3])  # batch x height x width x color

#Getting a tf.Tensor object's rank
r = tf.rank(my_image)
print(sess.run(r))

#Referring to tf.Tensor slices
####For a rank 0 tensor (a scalar), no indices are necessary, since it is already a single number.
####For a rank 1 tensor (a vector), passing a single index allows you to access a number:
my_vector = tf.Variable([1, 2, 3, 4], tf.int32)
my_scalar = my_vector[2]
####For tensors of rank 2 or higher, the situation is more interesting. For a tf.Tensor of rank 2, passing two numbers returns a scalar, as expected:
my_matrix = tf.Variable([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tf.int32)
my_scalar = my_matrix[1, 2]
####Passing a single number, however, returns a subvector of a matrix, as follows:
my_row_vector = my_matrix[2]
my_column_vector = my_matrix[:, 2]


#### SHAPE ####

#Getting a tf.Tensor object's shape
zeros = tf.zeros(my_matrix.shape[1]) #make a vector of zeros with the same size as the number of columns in a given matrix

#Changing the shape of a tf.Tensor object
rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])  # Reshape existing content into a 6x10 matrix
matrixB = tf.reshape(matrix, [3, -1])  #  Reshape existing content into a 3x20 matrix. -1 tells reshape to calculate the size of this dimension.
matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # Reshape existing content into a 4x3x5 tensor

# Note that the number of elements of the reshaped Tensors has to match the
# original number of elements. Therefore, the following example generates an
# error because no possible value for the last dimension will match the number
# of elements.
# yet_another = tf.reshape(matrixAlt, [13, 2, -1])  # ERROR!


#### DATA TYPES ####

# Cast a constant integer tensor into floating point.
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)


#### EVALUATING TENSORS ####

with sess.as_default():
    constant = tf.constant([1, 2, 3])
    tensor = constant * constant
    print(tensor.eval())

    p = tf.placeholder(tf.float32)
    t = p + 1.0
    # t.eval()  # This will fail, since the placeholder did not get a value.
    print(t.eval(feed_dict={p: 2.0}))  # This will succeed because we're feeding a value to the placeholder.


#### PRINTING TENSORS ####

with sess.as_default():
    t = tf.constant(2)
    print(t)  # This will print the symbolic tensor when the graph is being built. This tensor does not have a value in this context.
    tf.Print(t, [t])  # This does nothing
    t = tf.Print(t, [t])  # Here we are using the value returned by tf.Print
    result = t + 1  # Now when result is evaluated the value of `t` will be printed.
    result.eval()


########## VARIABLES ##########

#### CREATING A VARIABLE ####

#This creates a variable named "my_variable" which is a three-dimensional tensor with shape [1, 2, 3].
#This variable will, by default, have the dtype tf.float32 and its initial value will be randomized via tf.glorot_uniform_initializer.
my_variable = tf.get_variable("my_variable", [1, 2, 3])

#You may optionally specify the dtype and initializer to tf.get_variable. For example:
my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32, initializer=tf.zeros_initializer)

#You may initialize a tf.Variable to have the value of a tf.Tensor. For example:
other_variable = tf.get_variable("other_variable", dtype=tf.int32, initializer=tf.constant([23, 42]))

#### VARIABLE COLLECTIONS ####

#If you don't want a variable to be trainable, add it to the tf.GraphKeys.LOCAL_VARIABLES:
my_local = tf.get_variable("my_local", shape=(), collections=[tf.GraphKeys.LOCAL_VARIABLES])

#Alternatively, you can specify trainable=False as an argument to tf.get_variable:
my_non_trainable = tf.get_variable("my_non_trainable", shape=(), trainable=False)

# To add a variable (or any other object) to a collection after creating the variable, call tf.add_to_collection:
tf.add_to_collection("my_collection_name", my_local)

#And to retrieve a list of all the variables (or other objects) you've placed in a collection you can use:
tf.get_collection("my_collection_name")

#### DEVICE PLACEMENT ####

#You can place variables on particular devices.
#For example, the following snippet creates a variable named v and places it on the second GPU device:
# with tf.device("/device:GPU:1"):
#     v = tf.get_variable("v", [1])

#tf.train.replica_device_setter, which can automatically place variables in parameter servers.
# cluster_spec = {
#     "ps": ["ps0:2222", "ps1:2222"],
#     "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
# with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
#     v = tf.get_variable("v", shape=[20, 20])  # this variable is placed in the parameter server by the replica_device_setter


#### INITIALIZING VARIABLES ####

#Before you can use a variable, it must be initialized.
#To initialize all trainable variables in one go, before training starts, call tf.global_variables_initializer().
#This function returns a single operation responsible for initializing all variables in the tf.GraphKeys.GLOBAL_VARIABLES collection.
#Running this operation initializes all variables. For example:
sess.run(tf.global_variables_initializer())
# Now all variables are initialized.

#If you do need to initialize variables yourself, you can run the variable's initializer operation. For example:
sess.run(my_variable.initializer)

#You can also ask which variables have still not been initialized.
#For example, the following code prints the names of all variables which have not yet been initialized:
print(sess.run(tf.report_uninitialized_variables()))

#Note that by default tf.global_variables_initializer does not specify the order in which variables are initialized.
#Any time you use the value of a variable in a context in which not all variables are initialized
#(say, if you use a variable's value while initializing another variable), it is best to use variable.initialized_value() instead of variable:
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = tf.get_variable("w", initializer=v.initialized_value() + 1)

#### USING VARIABLES ####

va = tf.get_variable("va", shape=(), initializer=tf.zeros_initializer())
w = va + 1  # w is a tf.Tensor which is computed based on the value of v.
           # Any time a variable is used in an expression it gets automatically
           # converted to a tf.Tensor representing its value.

#To assign a value to a variable, use the methods assign, assign_add, and friends in the tf.Variable class.
var = tf.get_variable("var", shape=(), initializer=tf.zeros_initializer())
assignment = var.assign_add(1)
tf.global_variables_initializer().run(session=sess)
sess.run(assignment)  # or assignment.op.run(), or assignment.eval()

#To force a re-read of the value of a variable after something has happened, you can use tf.Variable.read_value:
var1 = tf.get_variable("var1", shape=(), initializer=tf.zeros_initializer())
assignment = var1.assign_add(1)
with tf.control_dependencies([assignment]):
    w = var1.read_value()  # w is guaranteed to reflect v's value after the assign_add operation.

#### SHARING VARIABLES ####

#TensorFlow supports two ways of sharing variables:
    # Explicitly passing tf.Variable objects around.
    # Implicitly wrapping tf.Variable objects within tf.variable_scope objects.

#Variable scopes allow you to control variable reuse when calling functions which implicitly create and use variables.
#They also allow you to name your variables in a hierarchical and understandable way.

#For example, let's say we write a function to create a convolutional / relu layer:
def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

#This function uses short names weights and biases, which is good for clarity.
#In a real model, however, we want many such convolutional layers, and calling this function repeatedly would not work
#Calling conv_relu in different scopes, however, clarifies that we want to create new variables:
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])

#If you do want the variables to be shared, you have two options. First, you can create a scope with the same name using reuse=True:
# with tf.variable_scope("model"):
#     output1 = my_image_filter(input1)
# with tf.variable_scope("model", reuse=True):
#     output2 = my_image_filter(input2)

#You can also call scope.reuse_variables() to trigger a reuse:
# with tf.variable_scope("model") as scope:
#     output1 = my_image_filter(input1)
#     scope.reuse_variables()
#     output2 = my_image_filter(input2)

#Since depending on exact string names of scopes can feel dangerous, it's also possible to initialize a variable scope based on another one:
# with tf.variable_scope("model") as scope:
#     output1 = my_image_filter(input1)
# with tf.variable_scope(scope, reuse=True):
#     output2 = my_image_filter(input2)


########## GRAPHS AND SESSIONS ##########

#### NAMING OPERATIONS ####

c_0 = tf.constant(0, name="c")  # => operation named "c"

# Already-used names will be "uniquified".
c_1 = tf.constant(2, name="c")  # => operation named "c_1"

# Name scopes add a prefix to all operations created in the same context.
with tf.name_scope("outer"):
    c_2 = tf.constant(2, name="c")  # => operation named "outer/c"

    # Name scopes nest like paths in a hierarchical file system.
    with tf.name_scope("inner"):
        c_3 = tf.constant(3, name="c")  # => operation named "outer/inner/c"

    # Exiting a name scope context will return to the previous prefix.
    c_4 = tf.constant(4, name="c")  # => operation named "outer/c_1"

    # Already-used name scopes will be "uniquified".
    with tf.name_scope("inner"):
        c_5 = tf.constant(5, name="c")  # => operation named "outer/inner_1/c"


#### PLACING OPERATIONS ON DIFFERENT DEVICES ####

# Operations created outside either context will run on the "best possible"
# device. For example, if you have a GPU and a CPU available, and the operation
# has a GPU implementation, TensorFlow will choose the GPU.
# weights = tf.random_normal(...)
#
# with tf.device("/device:CPU:0"):
#     # Operations created in this context will be pinned to the CPU.
#     img = tf.decode_jpeg(tf.read_file("img.jpg"))
#
# with tf.device("/device:GPU:0"):
#     # Operations created in this context will be pinned to the GPU.
#     result = tf.matmul(weights, img)

#If you are deploying TensorFlow in a typical distributed configuration,
#you might specify the job name and task ID to place variables on a task in the parameter server job ("/job:ps"),
#and the other operations on task in the worker job ("/job:worker"):
# with tf.device("/job:ps/task:0"):
#     weights_1 = tf.Variable(tf.truncated_normal([784, 100]))
#     biases_1 = tf.Variable(tf.zeroes([100]))
#
# with tf.device("/job:ps/task:1"):
#     weights_2 = tf.Variable(tf.truncated_normal([100, 10]))
#     biases_2 = tf.Variable(tf.zeroes([10]))
#
# with tf.device("/job:worker"):
#     layer_1 = tf.matmul(train_batch, weights_1) + biases_1
#     layer_2 = tf.matmul(train_batch, weights_2) + biases_2

#The tf.train.replica_device_setter API can be used with tf.device to place operations
#for data-parallel distributed training.
# with tf.device(tf.train.replica_device_setter(ps_tasks=3)):
#     # tf.Variable objects are, by default, placed on tasks in "/job:ps" in a round-robin fashion.
#     w_0 = tf.Variable(...)  # placed on "/job:ps/task:0"
#     b_0 = tf.Variable(...)  # placed on "/job:ps/task:1"
#     w_1 = tf.Variable(...)  # placed on "/job:ps/task:2"
#     b_1 = tf.Variable(...)  # placed on "/job:ps/task:0"
#
#     input_data = tf.placeholder(tf.float32)     # placed on "/job:worker"
#     layer_0 = tf.matmul(input_data, w_0) + b_0  # placed on "/job:worker"
#     layer_1 = tf.matmul(layer_0, w_1) + b_1     # placed on "/job:worker"

#### TENSOR-LIKE OBJECTS ####

# Many TensorFlow operations take one or more tf.Tensor objects as arguments.
# For example, tf.matmul takes two tf.Tensor objects, and tf.add_n takes a list of n tf.Tensor objects.
# For convenience, these functions will accept a tensor-like object in place of a tf.Tensor,
# and implicitly convert it to a tf.Tensor using the tf.convert_to_tensor method.
# Tensor-like objects include elements of the following types:
#
#     tf.Tensor
#     tf.Variable
#     numpy.ndarray
#     list (and lists of tensor-like objects)
#     Scalar Python types: bool, float, int, str
#
# You can register additional tensor-like types using tf.register_tensor_conversion_function.

#### EXECUTING A GRAPH IN A tf.Session ####

#### CREATING A tf.Session ####

# # Create a default in-process session.
# with tf.Session() as sess:
#   # ...
#
# # Create a remote session.
# with tf.Session("grpc://example.org:2222"):
#   # ...

#### USING tf.Session.run TO EXECUTE OPERATIONS ####

x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
output = tf.nn.softmax(y)
init_op = w.initializer

with tf.Session() as sess:
    # Run the initializer on `w`.
    sess.run(init_op)

    # Evaluate `output`. `sess.run(output)` will return a NumPy array containing
    # the result of the computation.
    print(sess.run(output))

    # Evaluate `y` and `output`. Note that `y` will only be computed once, and its
    # result used both to return `y_val` and as an input to the `tf.nn.softmax()`
    # op. Both `y_val` and `output_val` will be NumPy arrays.
    y_val, output_val = sess.run([y, output])


# Define a placeholder that expects a vector of three floating-point values, and a computation that depends on it.
x = tf.placeholder(tf.float32, shape=[3])
y = tf.square(x)

with tf.Session() as sess:
    # Feeding a value changes the result that is returned when you evaluate `y`.
    print(sess.run(y, {x: [1.0, 2.0, 3.0]}))  # => "[1.0, 4.0, 9.0]"
    print(sess.run(y, {x: [0.0, 0.0, 5.0]}))  # => "[0.0, 0.0, 25.0]"

    # Raises <a href="../api_docs/python/tf/errors/InvalidArgumentError"><code>tf.errors.InvalidArgumentError</code></a>,
    # because you must feed a value for a `tf.placeholder()` when evaluating a tensor that depends on it.
    # sess.run(y)

    # Raises `ValueError`, because the shape of `37.0` does not match the shape of placeholder `x`.
    # sess.run(y, {x: 37.0})


y = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
    # Define options for the `sess.run()` call.
    options = tf.RunOptions()
    options.output_partition_graphs = True
    options.trace_level = tf.RunOptions.FULL_TRACE

    # Define a container for the returned metadata.
    metadata = tf.RunMetadata()

    sess.run(y, options=options, run_metadata=metadata)

    # Print the subgraphs that executed on each device.
    print(metadata.partition_graphs)

    # Print the timings of each operation that executed.
    print(metadata.step_stats)


#### VISUALIZING YOUR GRAPH ####

# Build your graph.
# x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
# w = tf.Variable(tf.random_uniform([2, 2]))
# y = tf.matmul(x, w)
# # ...
# loss = ...
# train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)
#
# with tf.Session() as sess:
#     # `sess.graph` provides access to the graph used in a <a href="../api_docs/python/tf/Session"><code>tf.Session</code></a>.
#     writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)
#
#     # Perform your computation...
#     for i in range(1000):
#     sess.run(train_op)
#     # ...
#
#     writer.close()


#### PROGRAMMING WITH MULTIPLE GRAPHS ####

#You can install a different tf.Graph as the default graph, using the tf.Graph.as_default context manager:
g_1 = tf.Graph()
with g_1.as_default():
    # Operations created in this scope will be added to `g_1`.
    c = tf.constant("Node in g_1")

    # Sessions created in this scope will run operations from `g_1`.
    sess_1 = tf.Session()

g_2 = tf.Graph()
with g_2.as_default():
    # Operations created in this scope will be added to `g_2`.
    d = tf.constant("Node in g_2")

# Alternatively, you can pass a graph when constructing a <a href="../api_docs/python/tf/Session"><code>tf.Session</code></a>:
# `sess_2` will run operations from `g_2`.
sess_2 = tf.Session(graph=g_2)

assert c.graph is g_1
assert sess_1.graph is g_1

assert d.graph is g_2
assert sess_2.graph is g_2


#To inspect the current default graph, call tf.get_default_graph, which returns a tf.Graph object:
# Print all of the operations in the default graph.
g = tf.get_default_graph()
print(g.get_operations())


########## SAVE AND RESTORE ##########

#### SAVE VARIABLES ####

# # Create some variables.
# # v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
# # v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)
# #
# # inc_v1 = v1.assign(v1+1)
# # dec_v2 = v2.assign(v2-1)
# #
# # # Add an op to initialize the variables.
# # init_op = tf.global_variables_initializer()
# #
# # # Add ops to save and restore all the variables.
# # saver = tf.train.Saver()
# #
# # # Later, launch the model, initialize the variables, do some work, and save the
# # # variables to disk.
# # with tf.Session() as sess:
# #     sess.run(init_op)
# #     # Do some work with the model.
# #     inc_v1.op.run()
# #     dec_v2.op.run()
# #     # Save the variables to disk.
# #     save_path = saver.save(sess, "./checkpoints/model.ckpt")
# #     print("Model saved in path: %s" % save_path)

#### RESTORE VARIABLES ####

# tf.reset_default_graph()
#
# # Create some variables.
# v1 = tf.get_variable("v1", shape=[3])
# v2 = tf.get_variable("v2", shape=[5])
#
# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()
#
# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
# with tf.Session() as sess:
#   # Restore variables from disk.
#   saver.restore(sess, "/tmp/model.ckpt")
#   print("Model restored.")
#   # Check the values of the variables
#   print("v1 : %s" % v1.eval())
#   print("v2 : %s" % v2.eval())

#### CHOOSE VARIABLES TO SAVE AND RESTORE ####

# tf.reset_default_graph()
# # Create some variables.
# v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
# v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)
#
# # Add ops to save and restore only `v2` using the name "v2"
# saver = tf.train.Saver({"v2": v2})
#
# # Use the saver object normally after that.
# with tf.Session() as sess:
#     # Initialize v1 since the saver will not.
#     v1.initializer.run()
#     saver.restore(sess, "/tmp/model.ckpt")
#
#     print("v1 : %s" % v1.eval())
#     print("v2 : %s" % v2.eval())

# Notes:
#
#     You can create as many Saver objects as you want if you need to save and restore different subsets of the model variables.
#       The same variable can be listed in multiple saver objects; its value is only changed when the Saver.restore() method is run.
#
#     If you only restore a subset of the model variables at the start of a session, you have to run an initialize op for the other variables.
#       See tf.variables_initializer for more information.
#
#     To inspect the variables in a checkpoint, you can use the inspect_checkpoint library, particularly the print_tensors_in_checkpoint_file function.
#
#     By default, Saver uses the value of the tf.Variable.name property for each variable.
#       However, when you create a Saver object, you may optionally choose names for the variables in the checkpoint files.


#### INSPECT VARIABLES IN A CHECKPOINT ####

# # import the inspect_checkpoint library
# from tensorflow.python.tools import inspect_checkpoint as chkp
#
# # print all tensors in checkpoint file
# chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)
#
# # tensor_name:  v1
# # [ 1.  1.  1.]
# # tensor_name:  v2
# # [-1. -1. -1. -1. -1.]
#
# # print only tensor v1 in checkpoint file
# chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v1', all_tensors=False)
#
# # tensor_name:  v1
# # [ 1.  1.  1.]
#
# # print only tensor v2 in checkpoint file
# chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v2', all_tensors=False)
#
# # tensor_name:  v2
# # [-1. -1. -1. -1. -1.]


#### SAVE AND RESTORE MODELS ####

# Simple save

#The easiest way to create a SavedModel is to use the tf.saved_model.simple_save function:
# tf.savedModel.simple_save(session, export_dir, inputs={"x": x, "y": y}, outputs={"z": z})

# Manually build a SavedModel

#The tf.saved_model.builder.SavedModelBuilder class provides functionality to save multiple MetaGraphDefs.
# export_dir = ...
# ...
# builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
# with tf.Session(graph=tf.Graph()) as sess:
#   ...
#   builder.add_meta_graph_and_variables(sess,
#                                        [tag_constants.TRAINING],
#                                        signature_def_map=foo_signatures,
#                                        assets_collection=foo_assets,
#                                        strip_default_attrs=True)
# ...
# # Add a second MetaGraphDef for inference.
# with tf.Session(graph=tf.Graph()) as sess:
#   ...
#   builder.add_meta_graph([tag_constants.SERVING], strip_default_attrs=True)
# ...
# builder.save()

# Loading a SavedModel in Python

# The Python version of the SavedModel loader provides load and restore capability for a SavedModel.
# The load operation requires the following information:
#
#     The session in which to restore the graph definition and variables.
#     The tags used to identify the MetaGraphDef to load.
#     The location (directory) of the SavedModel.

# Upon a load, the subset of variables, assets, and signatures supplied as part of the specific MetaGraphDef will be restored into the supplied session.
# export_dir = ...
# ...
# with tf.Session(graph=tf.Graph()) as sess:
#   tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir)
#   ...



########## FEATURE COLUMNS ##########

#### NUMERIC COLUMNS ####

# Represent a tf.float64 scalar.
numeric_feature_column = tf.feature_column.numeric_column(key="SepalLength",
                                                          dtype=tf.float64)

# Represent a 10-element vector in which each cell contains a tf.float32.
vector_feature_column = tf.feature_column.numeric_column(key="Bowling",
                                                         shape=10)

# Represent a 10x5 matrix in which each cell contains a tf.float32.
matrix_feature_column = tf.feature_column.numeric_column(key="MyMatrix",
                                                         shape=[10,5])

#### BUCKETIZING COLUMNS ####

# First, convert the raw input to a numeric column.
numeric_feature_column = tf.feature_column.numeric_column("Year")

# Then, bucketize the numeric column on the years 1960, 1980, and 2000.
bucketized_feature_column = tf.feature_column.bucketized_column(
    source_column = numeric_feature_column,
    boundaries = [1960, 1980, 2000])


#### CATEGORICAL IDENTIY COLUMNS ####

# Create categorical output for an integer feature named "my_feature_b",
# The values of my_feature_b must be >= 0 and < num_buckets
# identity_feature_column = tf.feature_column.categorical_column_with_identity(
#     key='my_feature_b',
#     num_buckets=4) # Values [0, 4)
#
# # In order for the preceding call to work, the input_fn() must return
# # a dictionary containing 'my_feature_b' as a key. Furthermore, the values
# # assigned to 'my_feature_b' must belong to the set [0, 4).
# def input_fn():
#     ...
#     return ({ 'my_feature_a':[7, 9, 5, 2], 'my_feature_b':[3, 1, 2, 2] },
#             [Label_values])


#### CATEGORICAL VOCABULARY COLUMNS ####

# # Given input "feature_name_from_input_fn" which is a string,
# # create a categorical feature by mapping the input to one of
# # the elements in the vocabulary list.
# vocabulary_feature_column =
#     tf.feature_column.categorical_column_with_vocabulary_list(
#         key=feature_name_from_input_fn,
#         vocabulary_list=["kitchenware", "electronics", "sports"])


# # Given input "feature_name_from_input_fn" which is a string,
# # create a categorical feature to our model by mapping the input to one of
# # the elements in the vocabulary file
# vocabulary_feature_column =
#     tf.feature_column.categorical_column_with_vocabulary_file(
#         key=feature_name_from_input_fn,
#         vocabulary_file="product_class.txt",
#         vocabulary_size=3)


#### HASHED COLUMNS ####

# # pseudocode
# feature_id = hash(raw_feature) % hash_buckets_size

# hashed_feature_column =
#     tf.feature_column.categorical_column_with_hash_bucket(
#         key = "some_feature",
#         hash_buckets_size = 100) # The number of categories


#### CROSSED COLUMNS ####

# def make_dataset(latitude, longitude, labels):
#     assert latitude.shape == longitude.shape == labels.shape
#
#     features = {'latitude': latitude.flatten(),
#                 'longitude': longitude.flatten()}
#     labels=labels.flatten()
#
#     return tf.data.Dataset.from_tensor_slices((features, labels))
#
#
# # Bucketize the latitude and longitude using the `edges`
# latitude_bucket_fc = tf.feature_column.bucketized_column(
#     tf.feature_column.numeric_column('latitude'),
#     list(atlanta.latitude.edges))
#
# longitude_bucket_fc = tf.feature_column.bucketized_column(
#     tf.feature_column.numeric_column('longitude'),
#     list(atlanta.longitude.edges))
#
# # Cross the bucketized columns, using 5000 hash bins.
# crossed_lat_lon_fc = tf.feature_column.crossed_column(
#     [latitude_bucket_fc, longitude_bucket_fc], 5000)
#
# fc = [
#     latitude_bucket_fc,
#     longitude_bucket_fc,
#     crossed_lat_lon_fc]
#
# # Build and train the Estimator.
# est = tf.estimator.LinearRegressor(fc, ...)


#### INDICATOR COLUMNS ####

# categorical_column = ... # Create any type of categorical column.
#
# # Represent the categorical column as an indicator column.
# indicator_column = tf.feature_column.indicator_column(categorical_column)


#### EMBEDDING COLUMNS ####

# embedding_dimensions =  number_of_categories**0.25

# categorical_column = ... # Create any categorical column
#
# # Represent the categorical column as an embedding column.
# # This means creating an embedding vector lookup table with one element for each category.
# embedding_column = tf.feature_column.embedding_column(
#     categorical_column=categorical_column,
#     dimension=embedding_dimensions)


########## OPTIMIZER ##########

# # Create an optimizer with the desired parameters.
# opt = GradientDescentOptimizer(learning_rate=0.1)
# # Add Ops to the graph to minimize a cost by updating a list of variables.
# # "cost" is a Tensor, and the list of variables contains tf.Variable
# # objects.
# opt_op = opt.minimize(cost, var_list=<list of variables>)
#
# # Execute opt_op to do one step of training:
# opt_op.run()

#### PROCESSING GRADIENTS BEFORE APPLYING THEM ####

# # Create an optimizer.
# opt = GradientDescentOptimizer(learning_rate=0.1)
#
# # Compute the gradients for a list of variables.
# grads_and_vars = opt.compute_gradients(loss, <list of variables>)
#
# # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
# # need to the 'gradient' part, for example cap them, etc.
# capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]
#
# # Ask the optimizer to apply the capped gradients.
# opt.apply_gradients(capped_grads_and_vars)


# ####### READ DATASET FROM CSV FILE ########
#
#
# ds = tf.data.TextLineDataset(filepath).skip(1)
#
# # Metadata describing the text columns
# COLUMNS = ['fixed acidity', 'volatile acidity',
#            'citric acid', 'residual sugar',
#            'chlorides', 'free sulfur dioxide ', 'total sulfur dioxide',
#            'density', 'pH', 'sulphates', 'alcohol',
#            'quality']
# FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0],
#                   [0.0], [0.0], [0.0], [0.0],
#                   [0.0], [0.0], [0.0], [0]]
# def _parse_line(line):
#     # Decode the line into its fields
#     fields = tf.decode_csv(line, FIELD_DEFAULTS)
#
#     # Pack the result into a dictionary
#     features = dict(zip(COLUMNS,fields))
#
#     # Separate the label from the features
#     label = features.pop('quality')
#
#     return features, label
#
# ds = ds.map(_parse_line)
# print(ds)



