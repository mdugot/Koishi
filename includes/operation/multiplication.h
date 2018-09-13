#ifndef MULTIPLICATION_H
#define MULTIPLICATION_H
#include "operation/operation.h"

class Multiplication : public Operation {

    protected:

    public:
        Multiplication(Number *left, Number *right);
        float eval();
        float derivate(Variable *from);
};
#endif

