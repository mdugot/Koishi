#include "operation/reduction.h"

Reduction::Reduction(std::vector<Number*> vector) : vector(vector)
{
}

void Reduction::reinitGradient() {
    Number::reinitGradient();
    for (Number *n : vector) {
        n->reinitGradient();
    }
}

float Reduction::derivate(Variable *from) {
    float d = 0;
    for (unsigned int i = 0; i < vector.size(); i++) {
        d += vector[i]->derivate(from) * oneDerivative(i);
    }
    return d;
}

void Reduction::backpropagation(float gradient) {
    for (unsigned int i = 0; i < vector.size(); i++) {
        vector[i]->calculateGradient(gradient * oneDerivative(i));
    }
}
