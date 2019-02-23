#include "operation/multiplication.h"
#include "operation/inverse.h"
#include "operation/constant.h"

Multiplication::Multiplication(Number *left, Number *right) : Operation(left, right)
{
}

Division::Division(Number *left, Number *right) : Multiplication(left, new Inverse(right))
{
}

Negative::Negative(Number *number) : Multiplication(number, new Constant(-1))
{
}

FLOAT Multiplication::compute() {
    return left->eval() * right->eval();
}

FLOAT Multiplication::leftDerivative() {
    return right->eval();
}

FLOAT Multiplication::rightDerivative() {
    return left->eval();
}
