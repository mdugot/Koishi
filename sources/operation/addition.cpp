#include "operation/addition.h"

Addition::Addition(Number *left, Number *right) : Operation(left, right)
{
}

float Addition::eval() {
    return left->eval() + right->eval();
}

float Addition::derivate(Variable *from) {
   return left->derivate(from) + right->derivate(from);
}
