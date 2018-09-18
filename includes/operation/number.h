#ifndef NUMBER_H
#define NUMBER_H

#define EPSILON 0.0001

#include "utils.h"
#include "tensor/tensorInterface.h"
#include "tensor/tensorIndexException.h"

class Variable;

class Number : public TensorInterface {

    protected:
        float gradient;

    public:
        Number();
        virtual float eval() = 0;
        virtual float derivate(Variable *from) = 0;
        virtual void reinitGradient();
        float gradientChecking(Variable *from);
        void calculateGradient(float gradient = 1);
        virtual void backpropagation(float gradient) = 0;
        std::string toString();
        std::string toString(int margin);
        virtual TensorInterface* operator [](unsigned int idx);
        inline float getGradient() {return gradient;}
};
#endif
