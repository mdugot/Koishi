#include "operation/log.h"

Log::Log(Number *base) : Transformation(base)
{
}

FLOAT Log::compute() {
    return log(base->eval());
}

FLOAT Log::baseDerivative() {
    return 1/base->eval();
}

