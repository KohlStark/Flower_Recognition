#!/usr/bin/python
'''
Kohl Stark
Flower_Recognition
06/2019
Flower_Recognition.py
This python file contains the code used to effectively train and test a model created using Tensor Flow for
the first time. It is designed in such a way that it can be understood by people who do not code.

'''
#__________________________IMPORTS/WARNING_SUPRESSION_________________________________________
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import logging # needed to supress warnings
from tensorflow import contrib
import re
import matplotlib
from termcolor import colored
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) # needed to supress warnings
# needed to supress warnings
if type(tf.contrib) != type(tf): tf.contrib._warning = None
#__________________________TENSOR_FLOW_DATA_________________________________________

tf.enable_eager_execution()
print(colored("TensorFlow version:", 'blue'), tf.__version__, '\n')
#Check if eager execution is on
#print("Eager execution: {}".format(tf.executing_eagerly()))

#__________________________IMPORT_DATA_________________________________________

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url), origin=train_dataset_url)
print(colored("Local copy of the dataset file:", 'yellow'), train_dataset_fp, '\n')

#__________________________REPRESENT_DATA_________________________________________

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]
print("Hello! I am a machine learning algorithm that will be learning\nto identify flowers based on these features\n")
print(colored("Features:", 'green'), feature_names)
print("\nWhich have a labels like")
print(colored("Label:", 'green'), label_name)
print("\nThe data I will be studying looks like this!")
print(colored("Data:", 'red'), "5.9,3.0,4.2,1.5,1")
print("\nWhere the first four columns represent the features and the last column is it species")



#0: Iris setosa
#1: Iris versicolor
#2: Iris virginica
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
print("\nA 0 in the species column represents an:", class_names[0])
print("A 1 in the species column represents an:", class_names[1])
print("A 2 in the species column represents an:", class_names[2])
print("\nI will begin parsing the training data into suitable format for me to train on")

#__________________________CREATE_TF_DATASET_________________________________________

# Change the batch_size to set the number of examples stored in these feature arrays.
batch_size = 32

# Use the make_csv_dataset function to parse the data into a suitable format.
# Default behavior is to shuffle the data (shuffle=True, shuffle_buffer_size=10000)
train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_fp,
    batch_size, 
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)
# The num_epochs variable is the number of times to loop over the dataset collection. 
features, labels = next(iter(train_dataset))


#print(features)
# The make_csv_dataset function returns a tf.data.Dataset of (features, label) pairs,
# where features is a dictionary: {'feature_name': value}

# Features has feature_name = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

"""OrderedDict([('sepal_length',
              <tf.Tensor: id=58, shape=(32,), dtype=float32, numpy=
              array([4.4, 5.2, 5.4, 5. , 5.8, 6.2, 4.7, 5. , 5.2, 4.6, 5. , 5. , 7.7,
                     5.7, 5.5, 6.7, 5.8, 6.1, 6.4, 5.1, 4.8, 5.1, 6.3, 6.2, 5.8, 5.6,
                     7.2, 5.7, 6.4, 5. , 5. , 7. ], dtype=float32)>),
             ('sepal_width',
              <tf.Tensor: id=59, shape=(32,), dtype=float32, numpy=
              array([3.2, 3.4, 3.4, 3.6, 2.6, 2.2, 3.2, 3.2, 2.7, 3.6, 2. , 2.3, 3.8,
                     2.8, 2.4, 3.1, 2.7, 3. , 3.2, 3.5, 3.4, 2.5, 2.3, 2.8, 4. , 2.9,
                     3.2, 4.4, 2.8, 3. , 3.3, 3.2], dtype=float32)>),
             ('petal_length',
              <tf.Tensor: id=56, shape=(32,), dtype=float32, numpy=
              array([1.3, 1.4, 1.5, 1.4, 4. , 4.5, 1.3, 1.2, 3.9, 1. , 3.5, 3.3, 6.7,
                     4.5, 3.8, 4.4, 5.1, 4.9, 4.5, 1.4, 1.6, 3. , 4.4, 4.8, 1.2, 3.6,
                     6. , 1.5, 5.6, 1.6, 1.4, 4.7], dtype=float32)>),
             ('petal_width',
              <tf.Tensor: id=57, shape=(32,), dtype=float32, numpy=
              array([0.2, 0.2, 0.4, 0.2, 1.2, 1.5, 0.2, 0.2, 1.4, 0.2, 1. , 1. , 2.2,
                     1.3, 1.1, 1.4, 1.9, 1.8, 1.5, 0.3, 0.2, 1.1, 1.3, 1.8, 0.2, 1.3,
                     1.8, 0.4, 2.2, 0.2, 0.2, 1.4], dtype=float32)>)]) """

#__________________________PLOT_DATA_________________________________________

# Code to create plot based on petal and sepal length
choice = input("\nWould you like to see a plot of the training data? Enter Y/N\n")
if choice == ('Y' or 'y'):

  plt.scatter(features['petal_length'].numpy(),
            features['sepal_length'].numpy(),
            c=labels.numpy(),
            cmap='viridis')

  plt.xlabel("Petal length")
  plt.ylabel("Sepal length");
  plt.show()


#__________________________TRAINING_DATASET_________________________________________

#This function uses the tf.stack method
#which takes values from a list of tensors and creates a combined tensor at the specified dimension.
def pack_features_vector(features, labels):
  #"""Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

#Then use the tf.data.Dataset.map method to pack the
#features of each (features,label) pair into the training dataset:
train_dataset = train_dataset.map(pack_features_vector)


#The features element of the Dataset are now arrays with
#shape (batch_size, num_features). Let's look at the first few examples:
features, labels = next(iter(train_dataset))
#print(features)
"""tf.Tensor(
[[6.  2.2 5.  1.5]
 [6.  3.  4.8 1.8]
 [5.1 3.5 1.4 0.3]
 [7.7 3.8 6.7 2.2]
 [4.8 3.  1.4 0.1]], shape=(5, 4), dtype=float32)"""
time.sleep(5)
print("\nI have now parsed the data and am creating the model I will use to learn\nhow to identify each flower")

#__________________________CREATE_MODEL_________________________________________

#The tf.keras.Sequential model is a linear stack of layers.
#Its constructor takes a list of layer instances, in this case,
#two Dense layers with 10 nodes each, and an output layer
#with 3 nodes representing our label predictions.
#The first layer's input_shape parameter corresponds to the
#number of features from the dataset, and is required.

# 4->10->10->3
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])
time.sleep(3)
print("\nThe model (Nueral Network) has been created!\n")
#The activation function determines the output shape of each node in the layer.
#These non-linearities are important—without them the model would be equivalent to a single layer.
#There are many available activations, but ReLU is common for hidden layers.

#As a rule of thumb, increasing the number of hidden layers and neurons typically
#creates a more powerful model, which requires more data to train effectively.

#__________________________USE_MODEL_________________________________________

predictions = model(features)
#print("\n\n", predictions[:5])


#The vector of raw (non-normalized) predictions that a classification model generates,
#which is ordinarily then passed to a normalization function.
#If the model is solving a multi-class classification problem,
#logits typically become an input to the softmax function.
#The softmax function then generates a vector of (normalized) probabilities with
#one value for each possible class.

#To convert these logits to a probability for each class, use the softmax function:
#print("\n\n", tf.nn.softmax(predictions[:5]))


#Taking the tf.argmax across classes gives us the predicted class index.
#But, the model hasn't been trained yet, so these aren't good predictions.

#print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
#rint("    Labels: {}".format(labels))


#__________________________TRAINING_________________________________________

# If you learn too much about the training dataset,
#then the predictions only work for the data it has seen and will not be generalizable.
#This problem is called overfitting

#The Iris classification problem is an example of supervised machine learning:
#the model is trained from examples that contain labels.
#In unsupervised machine learning, the examples don't contain labels.
#Instead, the model typically finds patterns among the features.


#Both training and evaluation stages need to calculate the model's loss.
#This measures how off a model's predictions are from the desired label, in other words,
#how bad the model is performing. We want to minimize, or optimize, this value.


#Our model will calculate its loss using the tf.keras.losses.categorical_crossentropy function
#which takes the model's class probability predictions and the desired label, and
#returns the average loss across the examples.
time.sleep(3)
print("\nI will now create a loss function tell my how good my predictions\nare and use that as a way to keep track of my training progress")
def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


l = loss(model, features, labels)
#print("Loss test: {}".format(l))


#Use the tf.GradientTape context to calculate the gradients used to optimize our model.
#For more examples of this, see the eager execution guide.
# gradient = The vector of partial derivatives with respect to all of the independent variables.


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

#__________________________OPTIMIZER_________________________________________

#An optimizer applies the computed gradients to the model's variables to minimize the loss function.

#TensorFlow has many optimization algorithms available for training.
#This model uses the tf.train.GradientDescentOptimizer that implements the stochastic gradient descent (SGD)
#algorithm. The learning_rate sets the step size to take for each iteration down the hill.
#This is a hyperparameter that you'll commonly adjust to achieve better results.
time.sleep(3)
print("\nI will now create an optimizer to make adjustments\nto my neural network so that I may improve!")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

global_step = tf.Variable(0)

# We'll use this to calculate a single optimization step:
time.sleep(3)
print("\n\nHere are two examples of me taking small steps in learning by reducing my loss value!")

loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

print("Step: {},         Loss: {}".format(global_step.numpy(),
                                          loss(model, features, labels).numpy()))

#__________________________TRAINING_LOOP_________________________________________

#A training loop feeds the dataset examples into the model to help it make better predictions.
#The following code block sets up these training steps:

#1) Iterate each epoch. An epoch is one pass through the dataset.
#2) Within an epoch, iterate over each example in the training Dataset grabbing its features (x) and label (y).
#3) Using the example's features, make a prediction and compare it with the label.
#   Measure the inaccuracy of the prediction and use that to calculate the model's loss and gradients.
#4) Use an optimizer to update the model's variables.
#5) Keep track of some stats for visualization.
#6)Repeat for each epoch.

#The num_epochs variable is the number of times to loop over the dataset collection.
#Counter-intuitively, training a model longer does not guarantee a better model. FINE TUNED
time.sleep(3)
print("\nI will now begin training!")
print("\nThe workflow goes like this. One epoch represents one pass over the 120 samples.\nIn an epoch I look at all of the peices of flower data and make predictions on what\nspecies they are based on the data. I measure my inaccuracy for one epoch and use\nmy optimizer to update my models variables to learn! This process is repeated 201 times!\n\n\n")
print("I will now show you my training in increments of 50 epochs and you can watch\nmy loss go down and accuracy go up!")
time.sleep(5)
tfe = contrib.eager

# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201 # times to loop over dataset

for epoch in range(num_epochs):
  epoch_loss_avg = tfe.metrics.Mean()
  epoch_accuracy = tfe.metrics.Accuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables),
                              global_step)

    # Track progress
    epoch_loss_avg(loss_value)  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  # Print out every 50th epoch
  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

#__________________________VISUALIZE_LOSS_________________________________________
choice2 = input("\nWould you like to see a plot of the loss throughout training? Enter Y/N\n")
if choice2 == ('Y' or 'y'):
  fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
  fig.suptitle('Training Metrics')

  axes[0].set_ylabel("Loss", fontsize=14)
  axes[0].plot(train_loss_results)

  axes[1].set_ylabel("Accuracy", fontsize=14)
  axes[1].set_xlabel("Epoch", fontsize=14)
  axes[1].plot(train_accuracy_results);
  plt.show()

#__________________________TEST_DATASET_________________________________________
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)

test_dataset = tf.contrib.data.make_csv_dataset(
    test_fp,
    batch_size, 
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)

#__________________________EVAL_MODEL_ON_TEST_DATASET_________________________________________

test_accuracy = tfe.metrics.Accuracy()

for (x, y) in test_dataset:
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)
  #print("Predictions: {} \nActuals:     {}".format(prediction, y))

#print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
#print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
#print(tf.stack([y,prediction],axis=1))


#__________________________USED_TRAINED_MODEL_TO_MAKE_PREDICTIONS_________________________________________
print("\n\nYay! I've been now trained to recognize the difference between the flower species\n based on their features with considerable accuracy!")
print("Now I will look at flower data I have never seen before and guess its species!")
print("Below are 120 new samples, my predictions on what they are, and the correct answer!\n\n\n")
time.sleep(7)
file_contents = []
f = open('training_data.txt','r')
#file_contents.append(f.read())
#with open('passwords.txt', 'r') as f:
 #   file_contents = f.readlines()
test_data_actuals = []
for line in f:
    #print(line)
    temp = []
    temp = re.split(',', line)
    test_data_actuals.append(temp[-1].rstrip('\n'))
    del temp[-1]
    int_holder = []
    for i in temp:
        #print("i: ", i)
        int_holder.append(float(i))
        
    file_contents.append(int_holder)

#print ("GL: ", file_contents)
#print ("Actuals: ", test_data_actuals)

predict_dataset = tf.convert_to_tensor(file_contents)
#tf.convert_to_tensor([
#    [5.1, 3.3, 1.7, 0.5,],
#    [5.9, 3.0, 4.2, 1.5,],
#    [6.9, 3.1, 5.4, 2.1]
#])

predictions = model(predict_dataset)

#class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

counter = 0
fail_count = 0
for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("#{}! I predict: {} with ({:4.1f}%) accuracy!".format(counter+1, name, 100*p))
  #print ("Name:", name)
  #print ("Counter:", class_names[counter])
  if int(test_data_actuals[counter]) == 0:
      print("It was actually: {}".format(class_names[0]))
      if name == class_names[0]:
        print(colored("I was right!", 'green'))
      else:
        print(colored("\n\n Oh no I was wrong! \n\n", 'red'))
        fail_count = fail_count + 1
  elif int(test_data_actuals[counter]) == 1:
      print("It was actually: {}".format(class_names[1]))
      if name == class_names[1]:
        print(colored("I was right!", 'green'))
      else:
        print(colored("\n\n Oh no I was wrong! \n\n", 'red'))
        fail_count = fail_count + 1
  elif int(test_data_actuals[counter]) == 2:
      print("It was actually: {}".format(class_names[2]))
      if name == class_names[2]:
        print(colored("I was right!", 'green'))
      else:
        print(colored("\n\n Oh no I was wrong! \n\n", 'red'))
        fail_count = fail_count + 1
  counter = counter + 1

#print('counter: ', counter)
print('\n\nNumber of incorrect answers: ', fail_count)
middle = counter - fail_count
my_accuracy =  middle/counter
print("Overall accuracy: {:.3%}".format(my_accuracy))

  




