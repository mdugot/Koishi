#ifndef WRAPPERTOOLS_H
#define WRAPPERTOOLS_H

#ifdef PYTHON_WRAPPER
#include <boost/python.hpp>
#include "utils.h"
#include "initializer/initializer.h"
#include "initializer/feed.h"

using namespace boost::python;

template<class T>
std::vector<T> listToVector(list &l) {
    std::vector<T> v;
    for (unsigned int i = 0; i < len(l); ++i) {
        v.push_back(extract<T>(l[i]));
    }
    return v;
}

class InitializerWrapper {

    public:
        Initializer *initializer;
        InitializerWrapper(Initializer* initializer);
        virtual ~InitializerWrapper();
        void init();
};

class FeedWrapper : public InitializerWrapper {

    public:
        Feed *feeder;
        FeedWrapper(Feed* initializer);
        virtual ~FeedWrapper();
        void feed(list &values);
};

InitializerWrapper *getUniformInitializer(FLOAT min, FLOAT max);
FeedWrapper *getFeedInitializer(list &values);

#endif

#endif
