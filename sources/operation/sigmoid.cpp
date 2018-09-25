#include "operation/sigmoid.h"

Sigmoid::Sigmoid(Number *base) : Transformation(base)
{
}

FLOAT Sigmoid::eval() {
    return 1 / (1 + exp(-base->eval()));
}

FLOAT Sigmoid::baseDerivative() {
    FLOAT sig = eval();
    return sig * (1-sig);
}
