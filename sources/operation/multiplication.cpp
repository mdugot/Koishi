#include "operation/multiplication.h"

Multiplication::Multiplication(Number *left, Number *right) : Operation(left, right)
{
}

float Multiplication::eval() {
    return left->eval() * right->eval();
}

float Multiplication::leftDerivative() {
    return right->eval();
}

float Multiplication::rightDerivative() {
    return left->eval();
}
