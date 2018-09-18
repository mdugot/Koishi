#include "operation/pow.h"

Pow::Pow(Number *left, Number *right) : Operation(left, right)
{
}

float Pow::eval() {
    return pow(left->eval(), right->eval());
}

float Pow::leftDerivative() {
    float re = right->eval();
    float le = left->eval();
    float ld = re * pow(le, (re-1));
    return ld;
}

float Pow::rightDerivative() {
    float re = right->eval();
    float le = left->eval();
    float rd = log(le) * pow(le, re);
    return rd;
}

