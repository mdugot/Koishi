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
        float derivate(Variable *from);
        virtual void backpropagation(float gradient);
        virtual float baseDerivative() = 0;
};
#endif
