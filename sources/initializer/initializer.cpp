#include "initializer/initializer.h"
#include "operation/variable.h"

Initializer::Initializer()
{
}

Initializer::~Initializer()
{
}

void Initializer::add(Variable *v) {
    variables.push_back(v);
}
