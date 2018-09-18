#include "operation/addition.h"

Addition::Addition(Number *left, Number *right) : Operation(left, right)
{
}

float Addition::eval() {
    return left->eval() + right->eval();
}

float Addition::leftDerivative() {
    return 1;
}

float Addition::rightDerivative() {
    return 1;
}
