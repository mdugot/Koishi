#ifndef INITIALIZER_H
#define INITIALIZER_H
#include "utils.h"

class Variable;

class Initializer {
    
    protected:
        std::default_random_engine generator;
        std::list<Variable*> variables;

    public:
        static std::list<Initializer*> all;
        static void initializeAll();

        Initializer();
        virtual ~Initializer();
        virtual void init() = 0 ;
        void add(Variable *v);
        void remove(Variable *v);

};
#endif

