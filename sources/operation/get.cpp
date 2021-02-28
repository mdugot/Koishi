#include "operation/get.h"

Get::Get(std::vector<Number*> vector, Number* index) : index(index), vector(vector)
{
    for (Number *n : vector) {
        n->usedBy += 1;
    }
    index->usedBy += 1;
}

Get::~Get() {
    for (Number *n : vector) {
        n->unset();
    }
    index->unset();
}

void Get::reinitGradient() {
    Number::reinitGradient();
    for (Number *n : vector) {
        n->reinitGradient();
    }
}

FLOAT Get::compute() {
    unsigned int idx = (unsigned int)index->eval();
    if (idx >= vector.size()) {
        throw NumberException("index out of bounds");
    }
    return vector[idx]->eval();
}

FLOAT Get::derivate(Variable *from) {
    unsigned int idx = (unsigned int)index->eval();
    if (idx >= vector.size()) {
        throw NumberException("index out of bounds");
    }
    return vector[idx]->derivate(from);
}

void Get::backpropagation(FLOAT gradient) {
    unsigned int idx = (unsigned int)index->eval();
    if (idx >= vector.size()) {
        throw NumberException("index out of bounds");
    }
    vector[idx]->calculateGradient(gradient);
}
