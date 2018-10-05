#include "initializer/feed.h"
#include "operation/variable.h"

Feed::Feed() : Initializer()
{
    feed({0}, false);
}

Feed::Feed(std::vector<FLOAT> values) : Initializer()
{
    feed(values, false);
}

void Feed::feed(std::vector<FLOAT> values, bool doInit)
{
    if (values.size() == 0) {
        throw NumberException("Feed initializer must contain at least one value");
        
    }
    if (variables.size() % values.size() != 0)
        throw NumberException("Feed values must be a divisor of the number of variables to initialize");
    this->values = values;
    if (doInit)
        this->init();
}

Feed::~Feed()
{
}

void Feed::init() {
    unsigned int i = 0;
    unsigned int size = values.size();
    for (Variable *v : variables) {
        v->setValue(values[i%size]);
        i++;
    }
}
