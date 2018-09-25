#ifndef NUMBER_H
#define NUMBER_H

#define EPSILON 0.0000001
#define CHECKING_THRESHOLD 0.00001

#include "utils.h"

class Variable;
class Tensor;


class Number {

    protected:
        float gradient;

    public:
        static unsigned int count;

        unsigned int usedBy;

        Number();
        virtual ~Number();
        void unset();
        virtual float eval() = 0;
        virtual float derivate(Variable *from) = 0;
        virtual void reinitGradient();
        float gradientChecking(Variable *from);
        void checkAllGradient(std::string group);
        void calculateGradient(float gradient = 1);
        virtual void backpropagation(float gradient) = 0;
        std::string toString(bool printGradient = false);
        inline float getGradient() {return gradient;}
        bool equals(Number &number);
};
#endif
