#include "operation/number.h"
#include "operation/variable.h"

Number::Number() : TensorInterface(), gradient(0)
{
    isNumber = true;
}

void Number::calculateGradient(float gradient) {
    this->gradient += gradient;
    this->backpropagation(gradient);
}

float Number::gradientChecking(Variable *from) {
    float origin = from->getValue();

    from->setValue(origin-EPSILON);
    float r1 = eval();
    from->setValue(origin+EPSILON);
    float r2 = eval();
    from->setValue(origin);
    return (r2-r1) / (2*EPSILON);
}

void Number::reinitGradient() {
    gradient = 0;
}

TensorInterface* Number::operator[](unsigned int idx) {
    throw TensorIndexException("can not access element from 0 dimensional tensor", idx);
}

std::string Number::toString() {
    return std::to_string(eval());
}

std::string Number::toString(int margin) {
    (void)margin;
    return toString();
}
