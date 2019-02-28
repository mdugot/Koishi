#ifdef PYTHON_WRAPPER
#include "wrapper/wrapperTools.h"
#include "tensor/tensor.h"
#include "tensor/tensorException.h"
#include "initializer/uniform.h"
#include "initializer/feed.h"

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
    feeder->feed(getListShape(l), listToVector_deep<FLOAT>(l));
}

void FeedWrapper::feedSimple(FLOAT value) {
    feeder->feed({}, {value});
}

void FeedWrapper::feedNumpy(np::ndarray &a) {
    std::vector<unsigned int> dims = getNumpyShape(a);
    std::vector<FLOAT> values = numpyToVector(a);
    feeder->feed(dims, values);
}

InitializerWrapper *getUniformInitializer(FLOAT min, FLOAT max) {
    return new InitializerWrapper(new Uniform(min,max));
}

InitializerWrapper *getFillInitializer(FLOAT value) {
    return new InitializerWrapper(new Fill(value));
}

FeedWrapper *getFeedInitializer() {
    return new FeedWrapper(new Feed(std::vector<unsigned int>()));
}

FeedWrapper *getFeedInitializerFromList(list &l) {
    return new FeedWrapper(new Feed(listToVector<unsigned int>(l)));
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
    a = a.astype(np::dtype::get_builtin<float>());
    list l(a.reshape(make_tuple(-1)));
    std::vector<FLOAT> v;
    for (unsigned int i = 0; i < len(l); ++i) {
        v.push_back(float(extract<float_t>(l[i])));
    }
    return v;
}

std::vector<unsigned int> getNumpyShape(np::ndarray &a) {
    a = a.astype(np::dtype::get_builtin<float>());
    Py_intptr_t const * shape = a.get_shape();
    int ndim = a.get_nd();
    std::vector<unsigned int> v;
    for (int i = 0; i < ndim; ++i) {
        v.push_back(shape[i]);
    }
    return v;
}

void appendListShape(list &l, std::vector<unsigned int> &v) {
    v.push_back(len(l));

    unsigned int nextShape = 0;
    extract<object> objectExtractor(l[0]);
    object o=objectExtractor();
    std::string classname = extract<std::string>(o.attr("__class__").attr("__name__"));
    if (classname == "list") {
        list nextList = extract<list>(l[0]);
        nextShape = len(nextList);
        appendListShape(nextList, v);
    }
    for (unsigned int i = 0; i < len(l); ++i) {
        extract<object> objectExtractor(l[i]);
        object o=objectExtractor();
        std::string classname = extract<std::string>(o.attr("__class__").attr("__name__"));
        if (classname == "list" && len(extract<list>(l[i])) != nextShape) {
            throw TensorException("list has irregular shape");
        } else if (classname != "list" && nextShape != 0) {
            throw TensorException("list has irregular shape");
        }
    }
}

std::vector<unsigned int> getListShape(list &l) {
    std::vector<unsigned int> v;
    appendListShape(l, v);
    return v;
}

Tensor *newVariableWithGroup(list &dims, std::string group, InitializerWrapper &wrap) {
    return new Tensor(dims, group, wrap);
}

Tensor *newVariable(list &dims, InitializerWrapper &wrap) {
    return new Tensor(dims, "", wrap);
}

Tensor *newSimpleVariable(InitializerWrapper &wrap) {
    return new Tensor("", wrap);
}

Tensor *newSimpleVariableWithGroup(std::string group, InitializerWrapper &wrap) {
    return new Tensor(group, wrap);
}

Tensor *newVariableNumberWithGroup(std::string group, FLOAT value) {
    return new Tensor(group, value);
}

Tensor *newVariableNumber(FLOAT value) {
    return new Tensor("", value);
}

Tensor *newVariableNumpyWithGroup(std::string group, np::ndarray &a) {
    return new Tensor(group, a);
}

Tensor *newVariableNumpy(np::ndarray &a) {
    return new Tensor("", a);
}

Tensor *newVariableListWithGroup(std::string group, list &l) {
    return new Tensor(group, l);
}

Tensor *newVariableList(list &l) {
    return new Tensor("", l);
}


#endif
