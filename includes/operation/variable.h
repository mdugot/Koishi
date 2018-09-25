#ifndef VARIABLE_H
#define VARIABLE_H
#include "operation/constant.h"
#include "initializer/initializer.h"

class Variable : public Constant {
    
    private:
        std::string name;

    public:
        static std::multimap<std::string, Variable*> variables;

        Variable(std::string group, std::string name, Initializer &initializer);
        Variable(std::string group, std::string name, float value);
        inline std::string getName() {return name;}
        float derivate(Variable *from);

};
#endif
