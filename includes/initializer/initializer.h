#ifndef INITIALIZER_H
#define INITIALIZER_H
#include "utils.h"

class Variable;

class Initializer {
    
    protected:
        std::default_random_engine generator;
        std::vector<Variable*> variables;

    public:
        Initializer();
        virtual ~Initializer();
        virtual void init() = 0 ;
        void add(Variable *v);

};
#endif

