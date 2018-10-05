#ifdef PYTHON_WRAPPER
#include "wrapper/wrapperTools.h"
#include "tensor/tensor.h"
#include "tensor/tensorException.h"
#include "initializer/uniform.h"
#include "initializer/feed.h"
#include "operation/numberException.h"

InitializerWrapper::InitializerWrapper(Initializer *initializer) :initializer(initializer)
{
}

InitializerWrapper::~InitializerWrapper() {
    delete initializer;
}

void InitializerWrapper::init() {
    initializer->init();
}

FeedWrapper::FeedWrapper(Feed *feeder) :InitializerWrapper(feeder), feeder(feeder)
{
}

FeedWrapper::~FeedWrapper() {
}

void FeedWrapper::feed(list &l) {
    feeder->feed(listToVector<FLOAT>(l));
}

void FeedWrapper::feedSimple(FLOAT value) {
    feeder->feed({value});
}

void FeedWrapper::feedNumpy(np::ndarray &a) {
    std::vector<FLOAT> values = numpyToVector(a);
    feeder->feed(values);
}

InitializerWrapper *getUniformInitializer(FLOAT min, FLOAT max) {
    return new InitializerWrapper(new Uniform(min,max));
}

InitializerWrapper *getFillInitializer(FLOAT value) {
    return new InitializerWrapper(new Fill(value));
}

FeedWrapper *getFeedInitializer() {
    return new FeedWrapper(new Feed());
}

object Tensor::evalForPython() {
    if (dims.size() == 0)
        return object(content[0]->eval());
    list result;
    for (unsigned int i = 0; i < dims[0]; i++) {
        result.append(getTmp(i).evalForPython());
    }
    return result;
}

std::vector<FLOAT> numpyToVector(np::ndarray &a) {
    DEBUG << "numpy to vector\n";
    list l(a.reshape(make_tuple(-1)));
    std::vector<FLOAT> v;
    for (unsigned int i = 0; i < len(l); ++i) {
        v.push_back(float(extract<float_t>(l[i])));
    }
    DEBUG << "end\n";
    return v;
}

std::vector<unsigned int> getNumpyShape(np::ndarray &a) {
    DEBUG << "get numpy shape\n";
    Py_intptr_t const * shape = a.get_shape();
    int ndim = a.get_nd();
    std::vector<unsigned int> v;
    for (int i = 0; i < ndim; ++i) {
        v.push_back(shape[i]);
    }
    DEBUG << "end\n";
    return v;
}


#endif
