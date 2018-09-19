#include "tensor/tensorInterface.h"
#include "tensor/tensor.h"
#include "operation/number.h"


TensorInterface::~TensorInterface() {
}

std::ostream& operator<<(std::ostream& os, const TensorInterface& tf) {
    os << (TensorInterface*)&tf;
    return os;
}

std::ostream& operator<<(std::ostream& os, TensorInterface* tf) {
    os << "(" << (void*)tf << ")" << NL << tf->toString(0) << NL;
    return os;
}

bool operator==(TensorInterface &t1, TensorInterface &t2) {
    return t1.equals(t2);
}
