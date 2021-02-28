#ifndef GET_H
#define GET_H
#include "operation/number.h"

class Get : public Number {

    protected:
        std::vector<Number*> vector;
        Number* index;

    public:
        Get(std::vector<Number*> vector, Number* index);
        ~Get();
        virtual void reinitGradient();
        FLOAT derivate(Variable *from);
        virtual void backpropagation(FLOAT gradient);
        virtual FLOAT compute();
};

#endif
