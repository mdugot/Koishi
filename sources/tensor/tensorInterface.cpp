#include "tensor/tensorInterface.h"

std::ostream& operator<<(std::ostream& os, TensorInterface& tf) {
    os << tf.toString(0) << NL;
    return os;
}
