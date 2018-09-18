#include "operation/sigmoid.h"

Sigmoid::Sigmoid(Number *base) : Transformation(base)
{
}

float Sigmoid::eval() {
    return 1 / (1 + exp(-base->eval()));
}

float Sigmoid::baseDerivative() {
    float sig = eval();
    return sig * (1-sig);
}
