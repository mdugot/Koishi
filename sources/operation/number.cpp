#include "operation/number.h"
#include "operation/variable.h"
#include "tensor/tensor.h"

unsigned int Number::count = 0;

Number::Number() : gradient(0), usedBy(0)
{
    count += 1;
}

Number::~Number() {
    count -= 1;
}

void Number::unset() {
    usedBy -= 1;
    if (usedBy == 0)
        delete this;
}

void Number::calculateGradient(FLOAT gradient) {
    this->gradient += gradient;
    this->backpropagation(gradient);
}

FLOAT Number::gradientChecking(Variable *from) {
    FLOAT origin = from->getValue();

    from->setValue(origin-EPSILON);
    FLOAT r1 = eval();
    from->setValue(origin+EPSILON);
    FLOAT r2 = eval();
    from->setValue(origin);
    return (r2-r1) / (2*EPSILON);
}

void Number::reinitGradient() {
    gradient = 0;
}

std::string Number::toString(bool printGradient) {
    if (printGradient == false) {
        return std::to_string(eval());
    } else {
        return (std::to_string(eval()) + " (" + std::to_string(gradient) + ")");
    }
}

bool Number::equals(Number &number) {
    return eval() == number.eval();
}

void Number::checkAllGradient(std::string group) {
    reinitGradient();
    calculateGradient();
    auto bound = Variable::variables.equal_range(group);
    for (auto it = bound.first; it!=bound.second; ++it) {
        Variable *v = it->second;
        FLOAT checking = gradientChecking(v);
        FLOAT gradient = v->gradient;
        FLOAT diff = ABS(checking-gradient);
        bool error = diff > CHECKING_THRESHOLD;
        if (error) {
            DEBUG << RED << "[ERROR] check " + v->getName() + " : " + std::to_string(diff) << DEFAULT_COLOR << NL;
        } else {
            DEBUG << GREEN << "[OK] check " + v->getName() + " : " + std::to_string(diff) << DEFAULT_COLOR << NL;
        }
    }
}
