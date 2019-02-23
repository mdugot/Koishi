#include "operation/addition.h"
#include "operation/multiplication.h"

Substraction::Substraction(Number *left, Number *right) : Addition(left, new Negative(right))
{
}

Addition::Addition(Number *left, Number *right) : Operation(left, right)
{
}

FLOAT Addition::compute() {
    return left->eval() + right->eval();
}

FLOAT Addition::leftDerivative() {
    return 1;
}

FLOAT Addition::rightDerivative() {
    return 1;
}
