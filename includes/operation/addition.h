#ifndef ADDITION_H
#define ADDITION_H
#include "operation/operation.h"

class Addition : public Operation {

    protected:

    public:
        Addition(Number *left, Number *right);
        FLOAT compute();
        virtual FLOAT leftDerivative();
        virtual FLOAT rightDerivative();
};

class Substraction : public Addition {

    protected:

    public:
        Substraction(Number *left, Number *right);
};

#endif
