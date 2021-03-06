#ifndef MULTIPLICATION_H
#define MULTIPLICATION_H
#include "operation/operation.h"

class Multiplication : public Operation {

    protected:

    public:
        Multiplication(Number *left, Number *right);
        FLOAT compute();
        virtual FLOAT leftDerivative();
        virtual FLOAT rightDerivative();
};

class Division : public Multiplication {

    protected:

    public:
        Division(Number *left, Number *right);
};

class Negative : public Multiplication {

    protected:

    public:
        Negative(Number *number);
};

#endif

