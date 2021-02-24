#include "operation/exp.h"

Exp::Exp(Number *base) : Transformation(base)
{
}

FLOAT Exp::compute() {
    return exp(base->eval());
}

FLOAT Exp::baseDerivative() {
    return eval();
}

