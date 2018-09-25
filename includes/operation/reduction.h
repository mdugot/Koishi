#ifndef REDUCTION_H
#define REDUCTION_H
#include "operation/number.h"

class Reduction : public Number {

    protected:
        std::vector<Number*> vector;

    public:
        Reduction(std::vector<Number*> vector);
        ~Reduction();
        virtual void reinitGradient();
        float derivate(Variable *from);
        virtual void backpropagation(float gradient);
        virtual float oneDerivative(unsigned int idx) = 0;
};

#endif

