#include "operation/variable.h"

std::multimap<std::string, Variable*> Variable::variablesByGroup;
std::map<std::string, Variable*> Variable::variablesById;

void Variable::saveAll(std::string filename) {
    std::ofstream file;
    file.open(filename);
    if (!file.is_open())
        throw NumberException("can not open file '" + filename + "'");
    for (auto it=variablesById.begin(); it != variablesById.end(); ++it) {
        file << it->first + "," + std::to_string(it->second->getValue()) << NL;
    }
    file.close();
}

void Variable::save(std::string filename, std::string group) {
    std::ofstream file;
    file.open(filename);
    if (!file.is_open())
        throw NumberException("can not open file '" + filename + "'");
    auto bound = variablesByGroup.equal_range(group);
    for (auto it = bound.first; it!=bound.second; ++it) {
        file << it->second->getId() + "," + std::to_string(it->second->getValue()) << NL;
    }
    file.close();
}

void Variable::load(std::string filename) {
    std::ifstream file;
    file.open(filename);
    if (!file.is_open())
        throw NumberException("can not open file '" + filename + "'");
    std::string line;
    while(std::getline(file, line)) {
        size_t p = line.rfind(",");
        if (p == std::string::npos)
            throw NumberException("format error in the line '" + line + "'");
        std::string id = line.substr(0,p);
        FLOAT v = STRING_TO_FLOAT(line.substr(p+1));
        if (variablesById.count(id) == 0) {
            throw NumberException("variable '" + id + "' does not exist");
        }
        variablesById[id]->setValue(v);
    }
    file.close();
}

Variable::Variable(std::string group, std::string name, Initializer *initializer) : Variable(group, name, (FLOAT)0)
{
    this->initializer = initializer;
    if (initializer)
        initializer->add(this);
}

Variable::Variable(std::string group, std::string name, FLOAT v) : Constant(v), name(name), group(group), momentum(0), squaredGradientAverage(0) {
    initializer = NULL;
    std::string id = group + "/" + name;
    this->id = id;
//    DEBUG << "new variable : " << id << NL;
    if (variablesById.count(id) > 0) {
        throw NumberException("variable '" + id + "' already exists");
    }
    variablesById.insert(std::pair<std::string, Variable*>(id, this));
    variablesByGroup.insert(std::pair<std::string, Variable*>(group, this));
}

Variable::~Variable() {
    variablesById.erase(id);
    auto bound = variablesByGroup.equal_range(group);
    for (auto it = bound.first; it!=bound.second; ++it) {
        if (it->second == this) {
            variablesByGroup.erase(it);
            break;
        }
    }
    if (initializer)
        initializer->remove(this);
}


FLOAT Variable::derivate(Variable *from) {
    if (from == this)
        return 1;
    return 0;
}

void Variable::gradientDescent(FLOAT learningRate) {
    setValue(getValue() - (gradient*learningRate));
}

void Variable::momentumOptim(FLOAT learningRate, FLOAT coef) {
    momentum = coef*momentum - gradient;
    setValue(getValue() + (momentum*learningRate));
}

void Variable::RMSPropOptim(FLOAT learningRate, FLOAT coef) {
    squaredGradientAverage = coef*squaredGradientAverage + (1-coef)*(gradient*gradient);
    setValue(getValue() - (learningRate / (sqrt(squaredGradientAverage)+EPSILON) * gradient));
}

void Variable::adamOptim(FLOAT learningRate, FLOAT coef1, FLOAT coef2) {
    momentum = coef1*momentum + (1-coef1)*gradient;
    squaredGradientAverage = coef2*squaredGradientAverage + (1-coef2)*(gradient*gradient);
    setValue(getValue() - (learningRate * (momentum/(1-coef1)) / (sqrt((squaredGradientAverage/(1-coef2)))+EPSILON)));
}
