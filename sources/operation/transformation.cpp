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

FLOAT Transformation::derivate(Variable *from) {
    return base->derivate(from) * baseDerivative();
}

void Transformation::backpropagation(FLOAT gradient) {
    base->calculateGradient(gradient * baseDerivative());
}
