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
        FLOAT store;
        unsigned int localEvalCount;

    public:
        static unsigned int count;
        static unsigned int globalEvalCount;

        static void reinitAllGradient(std::string group);
        static void optimizeByGradientDescent(std::string group, FLOAT learningRate);
        static void optimizeByMomentum(std::string group, FLOAT learningRate, FLOAT momentumCoef);
        static void optimizeByRMSProp(std::string group, FLOAT learningRate, FLOAT RMSCoef);
        static void optimizeByAdam(std::string group, FLOAT learningRate, FLOAT momentumCoef, FLOAT RMSCoef);

        inline static void reinitAllGradientAll() {reinitAllGradient("");}
        inline static void optimizeByGradientDescentAll(FLOAT learningRate) {optimizeByGradientDescent("", learningRate);}
        inline static void optimizeByMomentumAll(FLOAT learningRate, FLOAT momentumCoef) {optimizeByMomentum("", learningRate, momentumCoef);}
        inline static void optimizeByRMSPropAll(FLOAT learningRate, FLOAT RMSCoef) {optimizeByRMSProp("", learningRate, RMSCoef);}
        inline static void optimizeByAdamAll(FLOAT learningRate, FLOAT momentumCoef, FLOAT RMSCoef) {optimizeByAdam("", learningRate, momentumCoef, RMSCoef);}

        unsigned int usedBy;

        Number();
        virtual ~Number();
        void unset();
        FLOAT fastEval();
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
