#include "operation/addition.h"

Addition::Addition(Number *left, Number *right) : Operation(left, right)
{
}

FLOAT Addition::eval() {
    return left->eval() + right->eval();
}

FLOAT Addition::leftDerivative() {
    return 1;
}

FLOAT Addition::rightDerivative() {
    return 1;
}
