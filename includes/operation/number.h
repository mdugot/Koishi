#ifndef NUMBER_H
#define NUMBER_H

#define EPSILON 0.00000001
#define CHECKING_THRESHOLD 0.001

#include "utils.h"
#include "numberException.h"

class Variable;
class Tensor;


class Number {

    protected:
        FLOAT gradient;

    public:
        static unsigned int count;

        unsigned int usedBy;

        Number();
        virtual ~Number();
        void unset();
        virtual FLOAT eval() = 0;
        virtual FLOAT derivate(Variable *from) = 0;
        virtual void reinitGradient();
        FLOAT gradientChecking(Variable *from);
        void checkAllGradient(std::string group);
        void calculateGradient(FLOAT gradient = 1);
        virtual void backpropagation(FLOAT gradient) = 0;
        std::string toString(bool printGradient = false);
        inline FLOAT getGradient() {return gradient;}
        bool equals(Number &number);
};
#endif
