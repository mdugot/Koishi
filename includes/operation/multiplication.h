#ifndef MULTIPLICATION_H
#define MULTIPLICATION_H
#include "operation/operation.h"

class Multiplication : public Operation {

    protected:

    public:
        Multiplication(Number *left, Number *right);
        FLOAT eval();
        virtual FLOAT leftDerivative();
        virtual FLOAT rightDerivative();
};
#endif

