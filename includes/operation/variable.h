#ifndef VARIABLE_H
#define VARIABLE_H
#include "operation/constant.h"
#include "initializer/initializer.h"

class Variable : public Constant {
    
    private:
        std::string name;
        std::string group;
        std::string id;
        Initializer *initializer;

    public:
        static std::multimap<std::string, Variable*> variablesByGroup;
        static std::map<std::string, Variable*> variablesById;

        static void save(std::string filename, std::string group);
        static void saveAll(std::string filename);
        static void load(std::string filename);

        Variable(std::string group, std::string name, Initializer &initializer);
        Variable(std::string group, std::string name, FLOAT value);
        ~Variable();
        inline std::string getName() {return name;}
        inline std::string getId() {return id;}
        FLOAT derivate(Variable *from);
        void descentUpdate(FLOAT learningRate);
        inline void setInitializer(Initializer* init) {initializer = init;}
        inline Initializer* getInitializer() {return initializer;}

};
#endif
