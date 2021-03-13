#include "operation/cosine.h"

Sinus::Sinus(Number *base) : Transformation(base)
{
}

FLOAT Sinus::compute() {
    return sin(base->eval());
}

FLOAT Sinus::baseDerivative() {
    return cos(base->eval());
}


Cosinus::Cosinus(Number *base) : Transformation(base)
{
}

FLOAT Cosinus::compute() {
    return cos(base->eval());
}

FLOAT Cosinus::baseDerivative() {
    return -sin(base->eval());
}
