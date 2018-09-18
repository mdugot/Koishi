#include "operation/operation.h"

Operation::Operation(Number *left, Number *right) : Number(), left(left), right(right)
{
}

void Operation::reinitGradient() {
    Number::reinitGradient();
    left->reinitGradient();
    right->reinitGradient();
}

float Operation::derivate(Variable *from) {
    return leftDerivative()*left->derivate(from) + rightDerivative()*right->derivate(from);
}

void Operation::backpropagation(float gradient) {
    left->calculateGradient(gradient * leftDerivative());
    right->calculateGradient(gradient * rightDerivative());
}
