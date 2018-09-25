#ifndef POW_H
#define POW_H
#include "operation/operation.h"

class Pow : public Operation {

    protected:

    public:
        Pow(Number *left, Number *right);
        FLOAT eval();
        virtual FLOAT leftDerivative();
        virtual FLOAT rightDerivative();
};
#endif

