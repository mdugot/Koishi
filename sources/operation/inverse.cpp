#include "operation/inverse.h"

Inverse::Inverse(Number *base) : Transformation(base)
{
}

float Inverse::eval() {
    return 1 / base->eval();
}

float Inverse::baseDerivative() {
    float r = base->eval();
    return -(1/(r*r));
}

