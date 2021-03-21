#ifdef PYTHON_WRAPPER
#include "wrapper/wrapperTools.h"
#include "tensor/tensor.h"
#include "tensor/tensorException.h"
#include "initializer/uniform.h"
#include "initializer/feed.h"


std::map<std::string, FeedWrapper*> FeedWrapper::dict = std::map<std::string, FeedWrapper*>();


InitializerWrapper::InitializerWrapper(Initializer *initializer) :initializer(initializer)
{
}

InitializerWrapper::~InitializerWrapper() {
    delete initializer;
}

void InitializerWrapper::init() {
    initializer->init();
}

FeedWrapper::FeedWrapper(std::string name, Feed *feeder) :InitializerWrapper(feeder), feeder(feeder), name(name)
{
    if (dict.count(name) > 0)
        throw TensorException("feeder named " + name + " already exist");
    dict[name] = this;
}

FeedWrapper::~FeedWrapper() {
    dict.erase(name);
}

void FeedWrapper::feed(list &l) {
    feeder->feed(getListShape(l), listToVector_deep<FLOAT>(l));
}

void FeedWrapper::feed(FLOAT value) {
    feeder->feed({}, {value});
}

void FeedWrapper::feed(np::ndarray &a) {
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

Tensor *getFeedInitializer(std::string name) {
    FeedWrapper* wrapper = new FeedWrapper(name, new Feed(std::vector<unsigned int>()));
    Tensor *tensor = new Tensor(name, *wrapper);
    return tensor;
}

Tensor *getFeedInitializerFromList(std::string name, list &l) {
    FeedWrapper* wrapper = new FeedWrapper(name, new Feed(listToVector<unsigned int>(l)));
    Tensor *tensor = new Tensor(name, *wrapper);
    return tensor;
}

void saveListGroups(std::string filename, list &l) {
    Variable::saveGroups(filename, listToVector<std::string>(l));
}

void feedKwargs(dict &feeds) {
    for (int i = 0; i < boost::python::len(feeds); i++) {
        std::string name(extract<const char*>(feeds.keys()[i]));
        if (FeedWrapper::dict.count(name) == 0) {
            throw TensorException("unknown feed " + name);
        }
        extract<object> objectExtractor(feeds.values()[i]);
        object o=objectExtractor();
        std::string classname = extract<std::string>(o.attr("__class__").attr("__name__"));
        if (classname == "int" || classname == "float") {
            FeedWrapper::dict[name]->feed(extract<FLOAT>(feeds.values()[i]));
        }
        else if (classname == "list") {
            list l = extract<list>(feeds.values()[i]);
            FeedWrapper::dict[name]->feed(l);
        }
        else if (classname == "ndarray") {
            np::ndarray a = extract<np::ndarray>(feeds.values()[i]);
            FeedWrapper::dict[name]->feed(a);
        }
        else {
            throw TensorException("unknown feed type " + classname);
        }
    }
}

object rawFeed(tuple args, dict kwargs) {
    if (boost::python::len(args) != 0) {
        throw TensorException("feed only accept keyed argument for feeding");
    }
    feedKwargs(kwargs);
    return object();
}

object Tensor::rawEval(tuple args, dict kwargs) {
    if (boost::python::len(args) != 1) {
        throw TensorException("eval only accept keyed argument for feeding");
    }
    Tensor& self = extract<Tensor&>(args[0]);
    return self.evalForPython(kwargs);
}

object Tensor::rawPropagation(tuple args, dict kwargs) {
    if (boost::python::len(args) != 1) {
        throw TensorException("backpropagation only accept keyed argument for feeding");
    }
    feedKwargs(kwargs);
    Tensor& self = extract<Tensor&>(args[0]);
    self.gradientUpdate();
    return object();
}

Tensor* Tensor::concatenate(boost::python::list &list) {
    if (boost::python::len(list) == 0) {
        throw TensorException("try to concatenate empty list");
    }
    std::vector<Tensor*> tensors;
    for (int i = 0; i < boost::python::len(list); i++) {
        extract<object> objectExtractor(list[i]);
        object o = objectExtractor();
        std::string classname = extract<std::string>(o.attr("__class__").attr("__name__"));
        if (classname != "Tensor") {
            throw TensorException("can only concatenate Tensor object");
        }
        Tensor& tensor = extract<Tensor&>(list[i]);
        tensors.push_back(&tensor);
    }
    std::vector<unsigned int> dims;
    std::vector<Number*> values;
    dims.push_back(tensors.size());
    for (int i = 0; i < tensors[0]->dims.size(); i++) {
        dims.push_back(tensors[0]->dims[i]);
    }
    for (int i = 0; i < tensors.size(); i++) {
        if (tensors[i]->dims.size() != tensors[0]->dims.size()) {
            throw TensorException("can only concatenate Tensor of same shape");
        }
        for (int j = 0; j < tensors[0]->dims.size(); j++) {
            if (tensors[i]->dims[j] != tensors[0]->dims[j]) {
                throw TensorException("can only concatenate Tensor of same shape");
            }
        }
        for (int j = 0; j < tensors[i]->len; j++) {
            values.push_back(tensors[i]->content[j]);
        }
    }
    return new Tensor(dims, values);
}

Tensor* Tensor::stack(boost::python::list &list) {
    if (boost::python::len(list) == 0) {
        throw TensorException("try to stack empty list");
    }
    std::vector<Tensor*> tensors;
    for (int i = 0; i < boost::python::len(list); i++) {
        extract<object> objectExtractor(list[i]);
        object o = objectExtractor();
        std::string classname = extract<std::string>(o.attr("__class__").attr("__name__"));
        if (classname != "Tensor") {
            throw TensorException("can only stack Tensor object");
        }
        Tensor& tensor = extract<Tensor&>(list[i]);
        tensors.push_back(&tensor);
    }
    std::vector<unsigned int> dims;
    std::vector<Number*> values;
    for (int i = 0; i < tensors[0]->dims.size(); i++) {
        dims.push_back(tensors[0]->dims[i]);
    }
    dims.push_back(tensors.size());
    for (int j = 0; j < tensors[0]->len; j++) {
        for (int i = 0; i < tensors.size(); i++) {
            if (i == 0) {
                if (tensors[i]->dims.size() != tensors[0]->dims.size()) {
                    throw TensorException("can only stack Tensor of same shape");
                }
                for (int j = 0; j < tensors[0]->dims.size(); j++) {
                    if (tensors[i]->dims[j] != tensors[0]->dims[j]) {
                        throw TensorException("can only stack Tensor of same shape");
                    }
                }
            }
            values.push_back(tensors[i]->content[j]);
        }
    }
    return new Tensor(dims, values);
}


object Tensor::evalForPython(dict &feeds) {
    feedKwargs(feeds);
    if (dims.size() == 0)
        return object(content[0]->eval());
    list result;
    for (unsigned int i = 0; i < dims[0]; i++) {
        dict empty = dict();
        result.append(getTmp(i).evalForPython(empty));
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
