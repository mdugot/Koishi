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
