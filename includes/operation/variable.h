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
        FLOAT momentum;
        FLOAT squaredGradientAverage;

    public:
        static std::multimap<std::string, Variable*> variablesByGroup;
        static std::map<std::string, Variable*> variablesById;

        static void save(std::string filename, std::string group);
        static void saveGroups(std::string filename, std::vector<std::string> groups);
        static void saveAll(std::string filename);
        static void load(std::string filename);

        Variable(std::string group, std::string name, Initializer *initializer);
        Variable(std::string group, std::string name, FLOAT value);
        ~Variable();
        inline std::string getName() {return name;}
        inline std::string getId() {return id;}
        FLOAT derivate(Variable *from);
        void gradientDescent(FLOAT learningRate);
        void momentumOptim(FLOAT learningRate, FLOAT coef);
        void RMSPropOptim(FLOAT learningRate, FLOAT coef);
        void adamOptim(FLOAT learningRate, FLOAT coef1, FLOAT coef2);
        inline void setInitializer(Initializer* init) {initializer = init;}
        inline Initializer* getInitializer() {return initializer;}

};
#endif
