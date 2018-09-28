#include "operation/variable.h"

std::multimap<std::string, Variable*> Variable::variablesByGroup;
std::map<std::string, Variable*> Variable::variablesById;

void Variable::save(std::string filename) {
    std::ofstream file;
    file.open(filename);
    if (!file.is_open())
        throw NumberException("can not open file '" + filename + "'");
    for (auto it=variablesById.begin(); it != variablesById.end(); ++it) {
        file << it->first + "," + std::to_string(it->second->getValue()) << NL;
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
        FLOAT value = STRING_TO_FLOAT(line.substr(p+1));
//        DEBUG << "id : " << key << ", value : " << value << NL;
        if (variablesById.count(id) == 0) {
            throw NumberException("variable '" + id + "' does not exist");
        }
        variablesById[id]->setValue(value);
    }
    file.close();
}

Variable::Variable(std::string group, std::string name, Initializer &initializer) : Variable(group, name, 0)
{
    initializer.add(this);
}

Variable::Variable(std::string group, std::string name, FLOAT value) : Constant(value), name(name), group(group) {
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
}


FLOAT Variable::derivate(Variable *from) {
    if (from == this)
        return 1;
    return 0;
}

void Variable::descentUpdate(FLOAT learningRate) {
    value -= gradient*learningRate;
}
