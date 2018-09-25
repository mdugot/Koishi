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
        FLOAT derivate(Variable *from);
        virtual void backpropagation(FLOAT gradient);
        virtual FLOAT oneDerivative(unsigned int idx) = 0;
};

#endif

