#ifndef NUMBER_H
#define NUMBER_H

#define EPSILON 0.0001

#include "utils.h"
#include "tensor/tensorIndexException.h"
#include "tensor/tensorException.h"

class Variable;
class Tensor;

class Number {

    protected:
        float gradient;

    public:
        unsigned int usedBy;

        Number();
        virtual ~Number();
        virtual float eval() = 0;
        virtual float derivate(Variable *from) = 0;
        virtual void reinitGradient();
        float gradientChecking(Variable *from);
        void calculateGradient(float gradient = 1);
        virtual void backpropagation(float gradient) = 0;
        std::string toString();
        inline float getGradient() {return gradient;}
        bool equals(Number &number);
};
#endif
