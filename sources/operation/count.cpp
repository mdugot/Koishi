#include "operation/count.h"

Count::Count(std::vector<Number*> vector) : Reduction(vector)
{
}

FLOAT Count::eval() {
    return (FLOAT)vector.size();
}

FLOAT Count::oneDerivative(unsigned int idx) {
    (void)idx;
    return 0;
}

