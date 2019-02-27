#ifndef LOG_H
#define LOG_H
#include "operation/transformation.h"

class Log : public Transformation {

    protected:

    public:
        Log(Number *base);
        FLOAT compute();
        virtual FLOAT baseDerivative();
};
#endif

