#ifndef INVERSE_H
#define INVERSE_H
#include "operation/transformation.h"

class Inverse : public Transformation {

    protected:

    public:
        Inverse(Number *base);
        FLOAT compute();
        virtual FLOAT baseDerivative();
};
#endif

