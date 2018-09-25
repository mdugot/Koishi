#ifndef INVERSE_H
#define INVERSE_H
#include "operation/transformation.h"

class Inverse : public Transformation {

    protected:

    public:
        Inverse(Number *base);
        FLOAT eval();
        virtual FLOAT baseDerivative();
};
#endif

