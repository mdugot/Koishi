#include "operation/pow.h"

Pow::Pow(Number *left, Number *right) : Operation(left, right)
{
}

FLOAT Pow::compute() {
    return pow(left->eval(), right->eval());
}

FLOAT Pow::leftDerivative() {
    FLOAT re = right->eval();
    FLOAT le = left->eval();
    FLOAT ld = re * pow(le, (re-1));
    return ld;
}

FLOAT Pow::rightDerivative() {
    FLOAT re = right->eval();
    FLOAT le = left->eval();
    FLOAT rd = log(le) * pow(le, re);
    return rd;
}

