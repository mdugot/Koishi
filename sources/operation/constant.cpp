#include "operation/constant.h"

Constant::Constant(float v) : Number()
{
    this->value = v;
}

float Constant::eval() {
    return value;
}

float Constant::derivate(Variable *from) {
    (void)from;
    return 0;
}

void Constant::backpropagation(float gradient) {
    (void)gradient;
}

