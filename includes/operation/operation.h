#ifndef OPERATION_H
#define OPERATION_H
#include "operation/number.h"

class Operation : public Number {

    protected:
        Number *left;
        Number *right;

    public:
        Operation(Number *left, Number *right);
        virtual void reinitGradient();
        float derivate(Variable *from);
        virtual void backpropagation(float gradient);
        virtual float leftDerivative() = 0;
        virtual float rightDerivative() = 0;
};

#endif
