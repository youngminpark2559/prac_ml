# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/Code_projects/TEMP_FOR_GITHUB/prac_ml/TensorFlow/Kidel && \
# rm e.l && python 02_ConvolutionalNeuralNetwork-OCR.py \
# 2>&1 | tee -a e.l && code e.l

# Code is edited from original code of
# https://github.com/Kidel/Deep-Learning-CNN-for-Image-Recognition/blob/master/02_ConvolutionalNeuralNetwork-OCR.ipynb

# ================================================================================
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
from datetime import timedelta

# ================================================================================
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# ================================================================================
from util import Util
u = Util()

# ================================================================================
# print("tf.__version__",tf.__version__)
# 1.13.1

# ================================================================================
# @ Configure neural network

# @ For convolutional layer 1
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# @ For convolutional layer 2
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# @ For fully-connected layer
fc_size = 128             # Number of neurons in fully-connected layer.

train_batch_size = 64

# ================================================================================
# @ Data Load

data = input_data.read_data_sets('data/MNIST/', one_hot=True)
# print("data",type(data))
# <class 'tensorflow.contrib.learn.python.learn.datasets.base.Datasets'>

# ================================================================================
# print("data.train.labels",data.train.labels.shape)
# (55000, 10)
# print("data.test.labels",data.test.labels.shape)
# (10000, 10)
# print("data.validation.labels",data.validation.labels.shape)
# (5000, 10)

# ================================================================================
test_lbls=data.test.labels
# print("test_lbls",test_lbls)
# print("test_lbls",test_lbls.shape)
# (10000, 10)

# ================================================================================
data.test.cls = np.argmax(data.test.labels, axis=1)
# print("data.test.cls",data.test.cls)
# [7 2 1 ... 4 5 6]

# print("data.test.cls",data.test.cls.shape)
# (10000,)

# ================================================================================
# c img_size: MNIST image size: (28,28)
img_size = 28

# c img_size_flat: Number of pixels
img_size_flat = img_size * img_size

# c img_shape: Image shape
img_shape = (img_size, img_size)

# c num_channels: gray scale
num_channels = 1

# c num_classes: 10 digits
num_classes = 10

# ================================================================================
def plot_images(images, cls_true, cls_pred=None): 
    u.plot_images(images=images, cls_true=cls_true, cls_pred=cls_pred, img_size=img_size, img_shape=img_shape)

# ================================================================================
te_imgs=data.test.images
# print("te_imgs",te_imgs.shape)
# (10000, 784)

# @ Sample test images
images = data.test.images[0:9]
# print("images",images.shape)
# (9, 784)

# @ True class of sample test images
cls_true = data.test.cls[0:9]
# print("cls_true",cls_true.shape)
# (9,)

plot_images(images=images, cls_true=cls_true)

# ================================================================================
# Function to create new weights data
def new_weights(shape):
    some_data=tf.truncated_normal(shape, stddev=0.05)
    # print("some_data",some_data)
    # Tensor("truncated_normal:0", shape=(10, 10), dtype=float32)
    return tf.Variable(some_data)
# new_weights((10,10))

# ================================================================================
# Function to create new biases data
def new_biases(length):
    const_data=tf.constant(0.05, shape=[length])
    return tf.Variable(const_data)

# ================================================================================
# Function to create new conv layer
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # c shape: [5,5,1,32]
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights (filters) using given shape
    weights = new_weights(shape=shape)
    
    # Create new biases, one for each filter
    biases = new_biases(length=num_filters)
    
    # ================================================================================
    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter is moved 2 pixels across the x- and y-axis of the image.
    # padding 'SAME': size of input = size of output
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    
    # ================================================================================
    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases
    
    # ================================================================================
    # pooling: down size images
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    
    # ================================================================================
    # adds some non-linearity to the formula 
    # allows the model to learn more complicated patterns
    layer = tf.nn.relu(layer)
    
    # ================================================================================
    # Normal case: max_pool(relu(x))
    # But relu(max_pool(x)) == max_pool(relu(x))
    # 75% more computationally efficient: relu(max_pool(x))
    
    # ================================================================================
    # resulting layer
    # filter-weights
    # Later you'll do: plot(filter-weights)
    return layer, weights

# ================================================================================
# Function to flatten layer
def flatten_layer(layer):

    layer_shape = layer.get_shape()

    # The shape of the input layer:
    # [num_images, img_height, img_width, num_channels]

    # The number of features: img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()
    
    # ================================================================================
    # Reshape the layer to [num_images, num_features]
    layer_flat = tf.reshape(layer, [-1, num_features])
    
    # ================================================================================
    # The shape of the flattened layer:
    # [num_images, img_height * img_width * num_channels]
    
    # ================================================================================
    # flattened layer
    # the number of features
    return layer_flat, num_features

# ================================================================================
# Function to create new FC layer
def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?
    
    # ================================================================================
    # Create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    
    # ================================================================================
    layer = tf.matmul(input, weights) + biases
    
    # ================================================================================
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

# ================================================================================
# c x: (n,flat_size)
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

# ================================================================================
# c x_image: (n,height,width,channel)
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

# ================================================================================
# c y_true: (n,10)
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')

# ================================================================================
# c y_true_cls: like 8
y_true_cls = tf.argmax(y_true, axis=1)

# ================================================================================
# @ Create conv1 layer

# print("x_image",x_image)
# Tensor("Reshape:0", shape=(?, 28, 28, 1), dtype=float32)
# print("num_channels",num_channels)
# 1
# print("filter_size1",filter_size1)
# 5
# print("num_filters1",num_filters1)
# 16

layer_conv1, weights_conv1 = new_conv_layer(
    input=x_image,num_input_channels=num_channels,filter_size=filter_size1,num_filters=num_filters1,use_pooling=True)

# print("layer_conv1",layer_conv1)
# Tensor("Relu:0", shape=(?, 14, 14, 16), dtype=float32)

# print("weights_conv1",weights_conv1)
# <tf.Variable 'Variable_1:0' shape=(5, 5, 1, 16) dtype=float32_ref>

# ================================================================================
# @ Create conv2 layer
layer_conv2, weights_conv2 = new_conv_layer(
  input=layer_conv1,num_input_channels=num_filters1,filter_size=filter_size2,num_filters=num_filters2,use_pooling=True)

# ================================================================================
# @ Flatten layer_conv2
layer_flat, num_features = flatten_layer(layer_conv2)

# ================================================================================
# @ Create FC1 layer
layer_fc1 = new_fc_layer(input=layer_flat,num_inputs=num_features,num_outputs=fc_size,use_relu=True)

# ================================================================================
# @ Create FC2 layer
layer_fc2 = new_fc_layer(input=layer_fc1,num_inputs=fc_size,num_outputs=num_classes,use_relu=False)

# ================================================================================
# @ Create y_pred node
y_pred = tf.nn.softmax(layer_fc2)

# @ Create y_pred_cls node
y_pred_cls = tf.argmax(y_pred, dimension=1)

# ================================================================================
# @ Create cost function node
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# ================================================================================
# @ Create optimizer node
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# ================================================================================
# @ Create perfomance measure node
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ================================================================================
# @ Run the network graph

session = tf.Session()
# session.run(tf.initialize_all_variables())
session.run(tf.global_variables_initializer())

# ================================================================================
total_iterations = 0

def optimize(num_iterations):
    # @ make total_iterations as global variable from the inside of function
    global total_iterations
    
    # ================================================================================
    start_time = time.time()
    
    # ================================================================================
    for i in range(total_iterations, total_iterations + num_iterations):
        
        # Get a batch of training examples.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        
        # ================================================================================
        # Put the batch into a dict with the proper names
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        
        # ================================================================================
        # Run the optimizer using this batch of training data.
        session.run(optimizer, feed_dict=feed_dict_train)
        
        # ================================================================================
        # Print status every 100 iterations.
        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))
    
    # ================================================================================
    # Update the total number of iterations performed.
    total_iterations += num_iterations
    
    # ================================================================================
    end_time = time.time()
    
    # ================================================================================
    # Difference between start and end-times.
    time_dif = end_time - start_time
    
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# ================================================================================
def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False): 
    # print("data",data)
    # Datasets(train=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x7f3ba42509b0>, validation=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x7f3ba4250a90>, test=<tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x7f3ba4250be0>)
    # print("data",type(data))
    # <class 'tensorflow.contrib.learn.python.learn.datasets.base.Datasets'>

    # print("x",x.shape)
    # (?, 784)
    # print("x",type(x))
    # <class 'tensorflow.python.framework.ops.Tensor'>
    # print("y_true",y_true.shape)
    # (?, 10)
    # print("y_true",type(y_true))
    # <class 'tensorflow.python.framework.ops.Tensor'>
    
    u.print_test_accuracy(
      session=session, data=data, x=x, y_true=y_true, y_pred_cls=y_pred_cls, num_classes=num_classes, 
      show_example_errors=show_example_errors, show_confusion_matrix=show_confusion_matrix)

# ================================================================================
# @ Performance before learning
print_test_accuracy()

# ================================================================================
# @ Performance after learning (1 optimization iteration)
optimize(num_iterations=1)
print_test_accuracy()

# ================================================================================
# @ Performance after learning (100 optimization iteration)
# We already performed 1 iteration above.
optimize(num_iterations=99)
print_test_accuracy(show_example_errors=True)

# ================================================================================
# @ Performance after learning (1000 optimization iteration)
# We performed 100 iterations above.
optimize(num_iterations=900)
print_test_accuracy(show_example_errors=True)

# ================================================================================
# @ Performance after learning (10000 optimization iteration)
# We performed 1000 iterations above.
# optimize(num_iterations=9000)
# print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)

# ================================================================================
# @ Visualize weights (conv filter) and feature maps

# Function to visualize conv filter
def plot_conv_weights(weights, input_channel=0):
    u.plot_conv_weights(session=session, weights=weights, input_channel=input_channel)

# ================================================================================
# @ Function to visualize conv's feature map
def plot_conv_layer(layer, image):
    u.plot_conv_layer(session=session, x=x, layer=layer, image=image)

# ================================================================================
# @ Function to visualize image with nearest interpolation (upsize)
def plot_image(image):
    u.plot_image(image=image, img_shape=img_shape)

# ================================================================================
image1 = data.test.images[0]
plot_image(image1)
# 7 number image

# ================================================================================
# @ Visualize filters in conv1 layer
plot_conv_weights(weights=weights_conv1)

# ================================================================================
# @ Feature map which is passed throught above filters of conv1 layer
plot_conv_layer(layer=layer_conv1, image=image1)

# ================================================================================
# @ Visualize filters in conv2 layer
plot_conv_weights(weights=weights_conv2, input_channel=0)

# ================================================================================
# @ Feature map which is passed throught above filters of conv1 layer
plot_conv_layer(layer=layer_conv2, image=image1)

# ================================================================================
# @ Close the session to release its resources.
session.close()
