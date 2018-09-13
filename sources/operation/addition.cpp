#include "operation/addition.h"

Addition::Addition(Number *left, Number *right) : Operation(left, right)
{
}

float Addition::eval() {
    return left->eval() + right->eval();
}
