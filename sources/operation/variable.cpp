#include "operation/variable.h"

Variable::Variable(float v) : Number()
{
    this->value = v;
}

float Variable::eval() {
    return value;
}

float Variable::derivate(Variable *from) {
    if (from == this)
        return 1;
    return 0;
}

void Variable::backpropagation(float gradient) {
    (void)gradient;
}
