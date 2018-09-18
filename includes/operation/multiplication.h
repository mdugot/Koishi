#ifndef MULTIPLICATION_H
#define MULTIPLICATION_H
#include "operation/operation.h"

class Multiplication : public Operation {

    protected:

    public:
        Multiplication(Number *left, Number *right);
        float eval();
        virtual float leftDerivative();
        virtual float rightDerivative();
};
#endif

