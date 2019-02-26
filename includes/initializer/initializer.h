#ifndef INITIALIZER_H
#define INITIALIZER_H
#include "utils.h"

class Variable;

class Initializer {
    
    protected:
        std::default_random_engine generator;
        std::list<Variable*> variables;
        std::vector<unsigned int> dims;

    public:
        static std::list<Initializer*> all;
        static void initializeAll();

        Initializer();
        Initializer(std::vector<unsigned int> dims);
        inline std::vector<unsigned int> getDims() {return dims;}
        virtual ~Initializer();
        virtual void init() = 0 ;
        void add(Variable *v);
        void remove(Variable *v);

};
#endif

