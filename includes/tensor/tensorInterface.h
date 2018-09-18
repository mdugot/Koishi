#ifndef TENSORINTERFACE_H
#define TENSORINTERFACE_H 
#include "utils.h"

class TensorInterface {

    private:

    public:
        bool isNumber;
        virtual std::string toString(int margin) = 0;
        virtual TensorInterface* operator [](unsigned int idx) = 0;
};

std::ostream& operator<<(std::ostream& os, TensorInterface& tf);
#endif

