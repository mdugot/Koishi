#ifndef TENSORINTERFACE_H
#define TENSORINTERFACE_H 
#include "utils.h"

class Number;
class Tensor;
class TensorInterface {

    protected:

    public:
        virtual ~TensorInterface();
        bool isNumber;
        virtual std::string toString(int margin) = 0;
        virtual TensorInterface& operator [](unsigned int idx) = 0;
        virtual int len() = 0;
        virtual Tensor shape() = 0;
        virtual TensorInterface* self() = 0;
        virtual bool equals(Tensor &tensor) = 0;
        virtual bool equals(Number &number) = 0;
        virtual bool equals(TensorInterface &tensor) = 0;

    friend bool operator==(const TensorInterface &t1, const TensorInterface&t2);
};

std::ostream& operator<<(std::ostream& os, const TensorInterface& tf);
std::ostream& operator<<(std::ostream& os, TensorInterface* tf);
bool operator==(TensorInterface &t1, TensorInterface &t2);
inline bool operator!=(TensorInterface &t1, TensorInterface &t2) {return !(t1 == t2);}

#endif
