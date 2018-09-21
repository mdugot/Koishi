#include "operation/number.h"
#include "operation/variable.h"
#include "tensor/tensor.h"

Number::Number() : gradient(0), usedBy(0)
{
}

Number::~Number() {
}

void Number::calculateGradient(float gradient) {
    this->gradient += gradient;
    this->backpropagation(gradient);
}

float Number::gradientChecking(Variable *from) {
    float origin = from->getValue();

    from->setValue(origin-EPSILON);
    float r1 = eval();
    from->setValue(origin+EPSILON);
    float r2 = eval();
    from->setValue(origin);
    return (r2-r1) / (2*EPSILON);
}

void Number::reinitGradient() {
    gradient = 0;
}

std::string Number::toString() {
    return std::to_string(eval());
}

bool Number::equals(Number &number) {
    return eval() == number.eval();
}

