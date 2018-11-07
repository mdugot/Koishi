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
    if (usedBy == 0) {
        delete this;
    }
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
    auto bound = Variable::variablesByGroup.equal_range(group);
    OUT << BLACK << HLIGHT_GREY << "Gradient checking :" << DEFAULT_COLOR << NL;
    unsigned int total = 0;
    unsigned int success = 0;
    for (auto it = bound.first; it!=bound.second; ++it) {
        Variable *v = it->second;
        FLOAT checking = gradientChecking(v);
        FLOAT gradient = v->gradient;
        FLOAT diff = ABS(checking-gradient);
        bool error = diff > CHECKING_THRESHOLD;
        total++;
        if (error) {
            OUT << RED << "[ERROR] check " + v->getName() + " : " + std::to_string(diff) << DEFAULT_COLOR << NL;
        } else {
            success++;
            OUT << GREEN << "[OK] check " + v->getName() + " : " + std::to_string(diff) << DEFAULT_COLOR << "\r";
        }
    }
    OUT << BLACK  << HLIGHT_GREY<<"Checking result : " << success << "/" << total << DEFAULT_COLOR << NL;
}

std::pair<
	std::multimap<std::string, Variable*>::iterator,
	std::multimap<std::string, Variable*>::iterator
> getVariableRange(std::string group) {
	if (group == "") {
		return std::pair<
			std::multimap<std::string, Variable*>::iterator,
			std::multimap<std::string, Variable*>::iterator
		>(
			Variable::variablesByGroup.begin(),
			Variable::variablesByGroup.end()
		);
	}
    return Variable::variablesByGroup.equal_range(group);
}

void Number::reinitAllGradient(std::string group) {
    auto bound = getVariableRange(group);
    for (auto it = bound.first; it!=bound.second; ++it) {
        Variable *v = it->second;
        v->gradient = 0;
    }
}

void Number::optimizeByGradientDescent(std::string group, FLOAT learningRate) {
    auto bound = getVariableRange(group);
    for (auto it = bound.first; it!=bound.second; ++it) {
        Variable *v = it->second;
        v->gradientDescent(learningRate);
        v->gradient = 0;
    }
}

void Number::optimizeByMomentum(std::string group, FLOAT learningRate, FLOAT momentumCoef) {
    auto bound = getVariableRange(group);
    for (auto it = bound.first; it!=bound.second; ++it) {
        Variable *v = it->second;
        v->momentumOptim(learningRate, momentumCoef);
        v->gradient = 0;
    }
}

void Number::optimizeByRMSProp(std::string group, FLOAT learningRate, FLOAT RMSCoef) {
    auto bound = getVariableRange(group);
    for (auto it = bound.first; it!=bound.second; ++it) {
        Variable *v = it->second;
        v->RMSPropOptim(learningRate, RMSCoef);
        v->gradient = 0;
    }
}

void Number::optimizeByAdam(std::string group, FLOAT learningRate, FLOAT momentumCoef, FLOAT RMSCoef) {
    auto bound = getVariableRange(group);
    for (auto it = bound.first; it!=bound.second; ++it) {
        Variable *v = it->second;
        v->adamOptim(learningRate, momentumCoef, RMSCoef);
        v->gradient = 0;
    }
}
