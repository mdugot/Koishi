#include "operation/percentile.h"

bool compareNumber(Number *n1, Number *n2) {
    return n1->eval() < n2->eval();
}

Percentile::Percentile(std::vector<Number*> vector, FLOAT percent) : Reduction(vector), sortedVector(vector), percent(percent)
{
    if (percent < 0 || percent > 100)
        throw NumberException("Percentile percent must be between 0 and 100");
}

void Percentile::sort() {
    std::sort(sortedVector.begin(), sortedVector.end(), compareNumber);
}

unsigned int Percentile::getPercentIndex() {
    if (percent <= 0)
        return 0;
    if (percent >= 100)
        return vector.size()-1;
    int r = (unsigned int)(ceil(percent/100*vector.size())-1);
    if (r <= 0)
        return 0;
    if ((unsigned int)r >= vector.size()-1)
        return vector.size()-1;
    return (unsigned int)r;
}

FLOAT Percentile::compute() {
    unsigned int index = getPercentIndex();
    sort();
    return sortedVector[index]->eval();
}

FLOAT Percentile::oneDerivative(unsigned int idx) {
    unsigned int index = getPercentIndex();
    sort();
    if (sortedVector[index] == vector[idx])
        return 1;
    return 0;
}

