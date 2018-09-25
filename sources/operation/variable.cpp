#include "operation/variable.h"

std::multimap<std::string, Variable*> Variable::variables;

Variable::Variable(std::string group, std::string name, Initializer &initializer) : Variable(group, name, 0)
{
    initializer.add(this);
}

Variable::Variable(std::string group, std::string name, FLOAT value) : Constant(value), name(name) {
    variables.insert(std::pair<std::string, Variable*>(group, this));
}

FLOAT Variable::derivate(Variable *from) {
    if (from == this)
        return 1;
    return 0;
}
