#ifndef WRAPPERTOOLS_H
#define WRAPPERTOOLS_H

#ifdef PYTHON_WRAPPER
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "utils.h"
#include "initializer/initializer.h"
#include "initializer/feed.h"
#include "initializer/fill.h"

using namespace boost::python;
namespace np = boost::python::numpy;

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
        void feedSimple(FLOAT value);
        void feedNumpy(np::ndarray &a);
};

InitializerWrapper *getUniformInitializer(FLOAT min, FLOAT max);
InitializerWrapper *getFillInitializer(FLOAT value);
FeedWrapper *getFeedInitializer(list &values);
FeedWrapper *getSimpleFeedInitializer(FLOAT value);
std::vector<FLOAT> numpyToVector(np::ndarray &a);
std::vector<unsigned int> getNumpyShape(np::ndarray &a);

#endif

#endif
