#include "operation/sum.h"

Sum::Sum(std::vector<Number*> vector) : Reduction(vector)
{
}

float Sum::eval() {
    float result = 0;
    for (unsigned int i = 0; i < vector.size(); i++) {
        result += vector[i]->eval();
    }
    return result;
}

float Sum::oneDerivative(unsigned int idx) {
    (void)idx;
    return 1;
}
