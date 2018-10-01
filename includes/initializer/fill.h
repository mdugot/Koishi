#ifndef FILL_H
#define FILL_H
#include "initializer/initializer.h"

class Fill : public Initializer {
    
    private:
        FLOAT value;


    public:
        Fill(FLOAT value);
        virtual ~Fill();
        void init();

};
#endif

