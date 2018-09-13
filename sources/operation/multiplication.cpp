#include "operation/multiplication.h"

Multiplication::Multiplication(Number *left, Number *right) : Operation(left, right)
{
}

float Multiplication::eval() {
    return left->eval() * right->eval();
}

float Multiplication::derivate(Variable *from) {
   return left->derivate(from)*right->eval() + right->derivate(from)*left->eval();
}
