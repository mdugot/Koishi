#ifndef PERCENTILE_H
#define PERCENTILE_H
#include "operation/reduction.h"

class Percentile : public Reduction {

    protected:
        std::vector<Number*> sortedVector;
        FLOAT percent;

        unsigned int getPercentIndex();
        void sort();

    public:
        Percentile(std::vector<Number*> vector, FLOAT percent);
        FLOAT compute();
        virtual FLOAT oneDerivative(unsigned int idx);
};

#endif

