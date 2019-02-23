#ifndef COUNT_H
#define COUNT_H
#include "operation/reduction.h"

class Count : public Reduction {

    protected:

    public:
        Count(std::vector<Number*> vector);
        FLOAT compute();
        virtual FLOAT oneDerivative(unsigned int idx);
};

#endif

