#ifndef NUMBER_H
#define NUMBER_H
#include "utils.h"

class Number {

    protected:

    public:
        Number();
        virtual float eval() = 0;
//        virtual float derivate(Variable *respectOf);
};
#endif
