#include "operation/constant.h"

Constant::Constant(FLOAT v) : Number()
{
    this->value = v;
}

FLOAT Constant::eval() {
    return value;
}

FLOAT Constant::derivate(Variable *from) {
    (void)from;
    return 0;
}

void Constant::backpropagation(FLOAT gradient) {
    (void)gradient;
}

