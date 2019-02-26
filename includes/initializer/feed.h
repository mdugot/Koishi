#ifndef FEED_H
#define FEED_H
#include "initializer/initializer.h"

class Feed : public Initializer {
    
    private:
        std::vector<FLOAT> values;


    public:
        Feed(std::vector<unsigned int> dims, std::vector<FLOAT> values);
        Feed();
        virtual ~Feed();
        void init();
        void feed(std::vector<unsigned int> dims, std::vector<FLOAT> values, bool doInit = true);

};
#endif

