#ifndef ADDITION_H
#define ADDITION_H
#include "operation/operation.h"

class Addition : public Operation {

    protected:

    public:
        Addition(Number *left, Number *right);
        float eval();
};
#endif
