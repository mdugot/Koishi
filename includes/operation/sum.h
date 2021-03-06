#ifndef SUM_H
#define SUM_H
#include "operation/reduction.h"

class Sum : public Reduction {

    protected:

    public:
        Sum(std::vector<Number*> vector);
        FLOAT compute();
        virtual FLOAT oneDerivative(unsigned int idx);
};

#endif

