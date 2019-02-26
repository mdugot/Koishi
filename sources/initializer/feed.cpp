#include "initializer/feed.h"
#include "operation/variable.h"

Feed::Feed() : Initializer()
{
}

Feed::Feed(std::vector<unsigned int> dims) : Initializer(dims)
{
}

void Feed::feed(std::vector<unsigned int> dims, std::vector<FLOAT> values, bool doInit)
{
    if (this->dims.size() > 0) {
        if (dims.size() != this->dims.size()) {
            throw NumberException("wrong shape of feed values");
        }
        for (unsigned int i = 0; i < this->dims.size(); i++) {
            if (dims[i] != this->dims[i]) {
                throw NumberException("wrong shape of feed values");
            }
        }
    } else if (values.size() > 1) {
        throw NumberException("wrong shape of feed values");
    }
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
        if (values.size() > 0) {
            v->setValue(values[i%size]);
        } else {
            v->setValue(0.0);
        }
        i++;
    }
}
