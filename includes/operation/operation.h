#ifndef OPERATION_H
#define OPERATION_H
#include "operation/number.h"

class Operation : public Number {

    protected:
        Number *left;
        Number *right;

    public:
        Operation(Number *left, Number *right);
        ~Operation();
        virtual void reinitGradient();
        FLOAT derivate(Variable *from);
        virtual void backpropagation(FLOAT gradient);
        virtual FLOAT leftDerivative() = 0;
        virtual FLOAT rightDerivative() = 0;
};

#endif
