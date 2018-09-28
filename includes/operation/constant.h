#ifndef CONSTANT_H
#define CONSTANT_H
#include "operation/number.h"

class Constant : public Number {
    
    protected:
        FLOAT value;

    public:
        Constant(FLOAT value);
        FLOAT eval();
        FLOAT derivate(Variable *from);
        virtual void backpropagation(FLOAT gradient);
        inline FLOAT getValue() {return value;}
        inline void setValue(FLOAT v) {value = v;}

};
#endif

