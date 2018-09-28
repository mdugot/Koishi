#include "initializer/feed.h"
#include "operation/variable.h"

Feed::Feed(std::vector<FLOAT> values) : Initializer()
{
    feed(values);
}

void Feed::feed(std::vector<FLOAT> values)
{
    if (values.size() == 0) {
        throw NumberException("Feed initializer must contain at least one value");
        
    }
    this->values = values;
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

