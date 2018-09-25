#include "operation/multiplication.h"

Multiplication::Multiplication(Number *left, Number *right) : Operation(left, right)
{
}

FLOAT Multiplication::eval() {
    return left->eval() * right->eval();
}

FLOAT Multiplication::leftDerivative() {
    return right->eval();
}

FLOAT Multiplication::rightDerivative() {
    return left->eval();
}
