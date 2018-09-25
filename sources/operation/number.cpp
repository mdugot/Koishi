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

void Number::calculateGradient(float gradient) {
    this->gradient += gradient;
    this->backpropagation(gradient);
}

float Number::gradientChecking(Variable *from) {
    float origin = from->getValue();

    DEBUG << "origin : " << origin << NL;
    from->setValue(origin-EPSILON);
    float r1 = eval();
    DEBUG << "r1 : " << origin << NL;
    from->setValue(origin+EPSILON);
    float r2 = eval();
    DEBUG << "r2 : " << origin << NL;
    from->setValue(origin);
    DEBUG << "gradient : " << ((r2-r1) / (2*EPSILON)) << NL;
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
        float checking = gradientChecking(v);
        float gradient = v->gradient;
        float diff = ABS(checking-gradient);
        bool error = diff > CHECKING_THRESHOLD;
        if (error) {
            DEBUG << RED << "[ERROR] check " + v->getName() + " : " + std::to_string(diff) << DEFAULT_COLOR << NL;
        } else {
            DEBUG << GREEN << "[OK] check " + v->getName() + " : " + std::to_string(diff) << DEFAULT_COLOR << NL;
        }
    }
}
