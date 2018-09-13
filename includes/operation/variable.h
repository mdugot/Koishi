#ifndef VARIABLE_H
#define VARIABLE_H
#include "operation/number.h"

class Variable : public Number {
    
    private:
        float value;

    public:
        Variable(float value);
        float eval();

};
#endif
