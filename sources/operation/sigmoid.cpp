#include "operation/sigmoid.h"

Sigmoid::Sigmoid(Number *base) : Transformation(base)
{
}

float Sigmoid::eval() {
    return 1 / (1 + exp(-base->eval()));
}

float Sigmoid::derivate(Variable *from) {
    float sig = eval();
    return base->derivate(from) * sig * (1-sig);
}
