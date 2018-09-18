#ifndef INVERSE_H
#define INVERSE_H
#include "operation/transformation.h"

class Inverse : public Transformation {

    protected:

    public:
        Inverse(Number *base);
        float eval();
        virtual float baseDerivative();
};
#endif

