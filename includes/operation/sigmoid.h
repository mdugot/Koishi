#ifndef SIGMOID_H
#define SIGMOID_H
#include "operation/transformation.h"

class Sigmoid : public Transformation {

    protected:

    public:
        Sigmoid(Number *base);
        FLOAT compute();
        virtual FLOAT baseDerivative();
};
#endif
