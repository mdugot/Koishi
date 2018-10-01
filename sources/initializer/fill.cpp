#include "initializer/fill.h"
#include "operation/variable.h"

Fill::Fill(FLOAT value) : Initializer(), value(value)
{
}

Fill::~Fill()
{
}

void Fill::init() {
    for (Variable *v : variables) {
        v->setValue(value);
    }
}

