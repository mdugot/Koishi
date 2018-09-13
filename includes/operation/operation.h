#ifndef OPERATION_H
#define OPERATION_H
#include "operation/number.h"

class Operation : public Number {

    protected:
        Number *left;
        Number *right;

    public:
        Operation(Number *left, Number *right);
};

#endif
