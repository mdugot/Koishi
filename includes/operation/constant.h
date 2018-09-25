#ifndef CONSTANT_H
#define CONSTANT_H
#include "operation/number.h"

class Constant : public Number {
    
    private:
        float value;

    public:
        Constant(float value);
        float eval();
        float derivate(Variable *from);
        virtual void backpropagation(float gradient);
        inline float getValue() {return value;}
        inline void setValue(float v) {value = v;}

};
#endif

