#ifndef POW_H
#define POW_H
#include "operation/operation.h"

class Pow : public Operation {

    protected:

    public:
        Pow(Number *left, Number *right);
        float eval();
        virtual float leftDerivative();
        virtual float rightDerivative();
};
#endif

