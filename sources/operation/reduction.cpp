#include "operation/reduction.h"

Reduction::Reduction(std::vector<Number*> vector) : vector(vector)
{
    for (Number *n : vector) {
        n->usedBy += 1;
    }
}

Reduction::~Reduction() {
    for (Number *n : vector) {
        n->unset();
    }
}

void Reduction::reinitGradient() {
    Number::reinitGradient();
    for (Number *n : vector) {
        n->reinitGradient();
    }
}

FLOAT Reduction::derivate(Variable *from) {
    FLOAT d = 0;
    for (unsigned int i = 0; i < vector.size(); i++) {
        d += vector[i]->derivate(from) * oneDerivative(i);
    }
    return d;
}

void Reduction::backpropagation(FLOAT gradient) {
    for (unsigned int i = 0; i < vector.size(); i++) {
        vector[i]->calculateGradient(gradient * oneDerivative(i));
    }
}
