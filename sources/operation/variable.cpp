#include "operation/variable.h"

Variable::Variable(float v) : Number()
{
    this->value = v;
}

float Variable::eval() {
    return value;
}

