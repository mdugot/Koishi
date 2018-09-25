#include "operation/transformation.h"

Transformation::Transformation(Number *base) : Number(), base(base)
{
    base->usedBy += 1;
}

Transformation::~Transformation() {
    base->unset();
}

void Transformation::reinitGradient() {
    Number::reinitGradient();
    base->reinitGradient();
}

float Transformation::derivate(Variable *from) {
    return base->derivate(from) * baseDerivative();
}

void Transformation::backpropagation(float gradient) {
    base->calculateGradient(gradient * baseDerivative());
}
