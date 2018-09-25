#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H
#include "operation/number.h"

class Transformation : public Number {

    protected:
        Number *base;

    public:
        Transformation(Number *base);
        ~Transformation();
        virtual void reinitGradient();
        FLOAT derivate(Variable *from);
        virtual void backpropagation(FLOAT gradient);
        virtual FLOAT baseDerivative() = 0;
};
#endif
