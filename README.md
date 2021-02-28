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

We will first load and preprocess the data. As some of the features of the kddcup99 dataset are string, we will use the method `get_dummies` of pandas to convert them into one hot vector. We will then regularize the dataset. The target will also be converted from string to an integer corresponding to the label.

```
data, target = fetch_kddcup99(return_X_y=True, as_frame=True, percent10=True, shuffle=True)  # load the dataset as a pandas dataframe
data = pandas.get_dummies(data, columns=['protocol_type', 'service', 'flag'])                # convert string features into one hot vector
labels = {label for label in target}                                                         # map target string to an integer   
labels_map = dict(zip(labels, range(len(labels))))
target = target.map(labels_map)
data = (data - data.mean(axis=0)) / (data.std(axis=0) + 0.000001)                            # regularize the dataset
data = data.to_numpy()                                                                       # convert to numpy arrays
target = target.to_numpy()
```
