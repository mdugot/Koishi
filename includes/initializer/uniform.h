#ifndef UNIFORM_H
#define UNIFORM_H
#include "initializer/initializer.h"

class Uniform : public Initializer {
    
    private:
        std::uniform_real_distribution<FLOAT> distribution;


    public:
        Uniform(FLOAT min, FLOAT max);
        virtual ~Uniform();
        void init();

};
#endif

