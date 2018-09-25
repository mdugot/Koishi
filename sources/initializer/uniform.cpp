#include "initializer/uniform.h"
#include "operation/variable.h"

Uniform::Uniform(FLOAT min, FLOAT max) : Initializer(), distribution(min,max)
{
}

Uniform::~Uniform()
{
}

void Uniform::init() {
    for (Variable *v : variables) {
        v->setValue(distribution(generator));
    }
}
