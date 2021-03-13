#ifndef COSINE_H
#define COSINE_H
#include "operation/transformation.h"

class Sinus : public Transformation {

    protected:

    public:
        Sinus(Number *base);
        FLOAT compute();
        virtual FLOAT baseDerivative();
};

class Cosinus : public Transformation {

    protected:

    public:
        Cosinus(Number *base);
        FLOAT compute();
        virtual FLOAT baseDerivative();
};
#endif

