#include "operation/inverse.h"

Inverse::Inverse(Number *base) : Transformation(base)
{
}

FLOAT Inverse::compute() {
    return 1 / base->eval();
}

FLOAT Inverse::baseDerivative() {
    FLOAT r = base->eval();
    return -(1/(r*r));
}

