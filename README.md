# KOISHI (computional graph for machine learning)

Koishi is a computational graph library for machine learning. It is written in C++ and wrapped into python3 module.

# Requirements

* ### C++11 Compiler
Source code is written in C++11.

 * ### CMake
*tested with version 3.19* <br/>
[CMake](https://cmake.org/) is used to make the library.

 * ### Python3
*tested with version 3.9* <br/>
Once compiled, the library will be wrapped into a python3 module.

 * ### Numpy
*tested with version 1.18* <br/>
Koishi is compatible with [Numpy](https://numpy.org/).
If missing, numpy will be automatically installed during the installation.

 * ### Boost
*tested with version 1.75* <br/>
[Boost](https://www.boost.org/) is used to handle the compatibility between c++ code and Python/numpy.

# Installation

Install the requirements, clone the repository move at the root of the repository and install the module with `setup.py`:

```
$> git clone https://github.com/mdugot/Koishi.git
$> cd Koishi
$> python3 setup.py install
```

To check the the module has been properly installed, move into an other directory and call the `greet()` method :

```
$> cd /tmp
$> python3 -c "import koishi; print(koishi.greet())"
KOISHI (computional graph for machine learning library)
```

# Example

In this example, we will use the koishi library to make a multilayer perceptron classifier for the [KDD-CUP-99](https://kdd.ics.uci.edu/databases/kddcup99/task.html) dataset. The goal of this dataset is to detect network intrusion from unauthorized users by distinguig the normal connections from the various type of intrusions or attacks. Each data has 41 features and belong to one of 23 classes.

Additionaly to koishi and numpy, this example will requires the following python modules :
 * **pandas** to preprocess the data
 * **sklearn** to load the kddcup99 dataset
 * **matplotlib** to plot a summary of the training

```
import koishi
import numpy as np
import pandas
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

We will first load and preprocess the data. As some of the features of the kddcup99 dataset are string, we will use the method `get_dummies` of pandas to convert them into one hot vector. We will then regularize the dataset. The target will also be converted from string to an integer corresponding to the label. The kddcup99 is a very big datasets, we will only load 10% of it and it will still be big enough to train with less than one epoch without having to split it into a test set and a train set.

```
data, target = fetch_kddcup99(return_X_y=True, as_frame=True, percent10=True, shuffle=True)  # load 10% of the dataset as a pandas dataframe
data = pandas.get_dummies(data, columns=['protocol_type', 'service', 'flag'])                # convert string features into one hot vector
labels = {label for label in target}                                                         # map target string to an integer   
labels_map = dict(zip(labels, range(len(labels))))
target = target.map(labels_map)
data = (data - data.mean(axis=0)) / (data.std(axis=0) + 0.000001)                            # regularize the dataset
data = data.to_numpy()                                                                       # convert to numpy arrays
target = target.to_numpy()
```

Let's define the hyperparameters of our model and for the optimizer. The model will be a simple MLP with one hidden layer. The optimizer will be the adam optimizer.

```
nfeatures = int(data.shape[1])
nclasses = int(target.max() + 1)
hidden_layer = 128
batch_size = 100

# Paremeters for the Adam optimizer
learning_rate = 0.0002
momentum = 0.9
rms = 0.9
```

Let's create a `koishi.uniformInitializer` to initialize the parameters of our model and some `koishi.feedInitializer` to feed the data to the model.

```
feed_inputs = koishi.feedInitializer([batch_size, nfeatures])
feed_labels = koishi.feedInitializer([batch_size])
init = koishi.uniformInitializer(-0.1,0.1)
```

Let's create the variables of our models. The variables corresping to the data feeded to the model will be initialized by the feed initializer while the learnable variables (the weights and the bias) will be initialized by the uniform initializer.

```
inputs = koishi.Variable('inputs', feed_inputs)
labels = koishi.Variable('labels', feed_labels)

w1 = koishi.Variable([nfeatures, hidden_layer], 'param', init)  # weigths of the first layer
w2 = koishi.Variable([hidden_layer, nclasses], 'param', init)   # weights of the second layer
b1 = koishi.Variable([hidden_layer], 'param', init)             # bias of the first layer
b2 = koishi.Variable([nclasses], 'param', init)                 # bias of the second layer
```

We can now create the computational graph of our model. We will split it into two parts, the forward function corresponding to the MLP with a final softmax activation to make some predictions based on the inputs and a cross entropy loss to train the model based on the outputs of the MLP and the true targets.

```
forward = inputs.matmul(w1).add(b1).sigmoid().matmul(w2).add(b2).softmax(1)
loss = forward.get(labels).log().negative().mean()
```

