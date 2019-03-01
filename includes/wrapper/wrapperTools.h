#ifndef WRAPPERTOOLS_H
#define WRAPPERTOOLS_H

#ifdef PYTHON_WRAPPER
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "utils.h"
#include "initializer/initializer.h"
#include "initializer/feed.h"
#include "initializer/fill.h"
#include "operation/numberException.h"
#include "operation/variable.h"

class TensorException;
class Tensor;

using namespace boost::python;
namespace np = boost::python::numpy;

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
FeedWrapper *getFeedInitializer();
FeedWrapper *getFeedInitializerFromList(list &l);
std::vector<FLOAT> numpyToVector(np::ndarray &a);
std::vector<unsigned int> getNumpyShape(np::ndarray &a);
std::vector<unsigned int> getListShape(list &l);

Tensor *newVariableNumpy(np::ndarray &a);
Tensor *newVariableNumpyWithGroup(std::string group, np::ndarray &a);

Tensor *newVariableList(list &l);
Tensor *newVariableListWithGroup(std::string group, list &l);

Tensor *newVariableNumber(FLOAT value);
Tensor *newVariableNumberWithGroup(std::string group, FLOAT value);

Tensor *newVariable(list &dims, InitializerWrapper &wrap);
Tensor *newVariableWithGroup(list &dims, std::string group, InitializerWrapper &wrap);
Tensor *newSimpleVariable(InitializerWrapper &wrap);
Tensor *newSimpleVariableWithGroup(std::string group, InitializerWrapper &wrap);

void saveListGroups(std::string filename, list &l);

template<class T>
void appendList(list &l, std::vector<T> &v) {
    for (unsigned int i = 0; i < len(l); ++i) {
        extract<object> objectExtractor(l[i]);
        object o=objectExtractor();
        std::string classname = extract<std::string>(o.attr("__class__").attr("__name__"));
        if (classname == "list") {
            list l2 = extract<list>(l[i]);
            appendList<T>(l2, v);
        } else {
            v.push_back(extract<T>(l[i]));
        }
    }
}

template<class T>
std::vector<T> listToVector_deep(list &l) {
    std::vector<T> v;
    appendList<T>(l, v);
    return v;
}

template<class T>
std::vector<T> listToVector(list &l) {
    std::vector<T> v;
    appendList<T>(l, v);
    std::vector<unsigned int> dims = getListShape(l);
    if (dims.size() != 1)
        throw NumberException("one dimensional list is expected");
    return v;
}

#endif

#endif
