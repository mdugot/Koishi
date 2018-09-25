#include "operation/operation.h"

Operation::Operation(Number *left, Number *right) : Number(), left(left), right(right)
{
    left->usedBy += 1;
    right->usedBy += 1;
}

Operation::~Operation() {
    left->unset();
    right->unset();
}

void Operation::reinitGradient() {
    Number::reinitGradient();
    left->reinitGradient();
    right->reinitGradient();
}

FLOAT Operation::derivate(Variable *from) {
    return leftDerivative()*left->derivate(from) + rightDerivative()*right->derivate(from);
}

void Operation::backpropagation(FLOAT gradient) {
    left->calculateGradient(gradient * leftDerivative());
    right->calculateGradient(gradient * rightDerivative());
}
