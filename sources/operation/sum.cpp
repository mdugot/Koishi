#include "operation/percentile.h"

bool compareNumber(Number *n1, Number *n2) {
    return n1->eval() < n2->eval();
}

Percentile::Percentile(std::vector<Number*> vector, FLOAT percent) : Reduction(vector), percent(percent)
{
    if (percent < 0 || percent > 100)
        throw NumberException("Percentile percent must be between 0 and 100", this);
}

std::map<std::pair<FLOAT, unsigned int>> Percentile::sortedEval() {
    std::list<Number*> results;
    for (unsigned int i = 0; i < vector.size(); i++) {
        results.push_back(vector[i]);
    }
    results.sort(compareNumber);
    return results;
}

unsigned int Percentile::getPercentIndex() {
    if (percent <= 0)
        return 0;
    if (percent >= 100)
        return vector.size()-1;
    return (unsigned int)(ceil(percent/100*vector.size())-1);
}

FLOAT Percentile::eval() {
    FLOAT result = 0;
    for (unsigned int i = 0; i < vector.size(); i++) {
        result += vector[i]->eval();
    }
    return result;
}

FLOAT Percentile::oneDerivative(unsigned int idx) {
    (void)idx;
    return 1;
}
