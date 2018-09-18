#ifndef VARIABLE_H
#define VARIABLE_H
#include "operation/number.h"

class Variable : public Number {
    
    private:
        float value;

    public:
        Variable(float value);
        float eval();
        float derivate(Variable *from);
        virtual void backpropagation(float gradient);
        inline float getValue() {return value;}
        inline void setValue(float v) {value = v;}

};
#endif
