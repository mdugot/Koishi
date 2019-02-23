#ifndef CONSTANT_H
#define CONSTANT_H
#include "operation/number.h"

class Constant : public Number {

    private:
        FLOAT value;

    public:
        Constant(FLOAT value);
        FLOAT compute();
        FLOAT derivate(Variable *from);
        virtual void backpropagation(FLOAT gradient);
        inline FLOAT getValue() {return value;}
        inline void setValue(FLOAT v) {Number::globalPhase += 1; value = v;}
};
#endif

