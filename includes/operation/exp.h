#ifndef EXP_H
#define EXP_H
#include "operation/transformation.h"

class Exp : public Transformation {

    protected:

    public:
        Exp(Number *base);
        FLOAT compute();
        virtual FLOAT baseDerivative();
};
#endif
