#ifdef PYTHON_WRAPPER
#include "tensor/tensor.h"
#include "operation/variable.h"

char const* greet()
{
   return "KOISHI (computional graph for machine learning library)";
}

BOOST_PYTHON_MODULE(koishi)
{
    Py_Initialize();
    np::initialize();

    def("greet", greet);

    def("initializeAll", &Initializer::initializeAll);
    def("save", &Variable::save);
    def("save", saveListGroups);
    def("saveAll", &Variable::saveAll);
    def("load", &Variable::load);

    def("Feeder", getFeedInitializer, return_value_policy<manage_new_object>());
    def("Feeder", getFeedInitializerFromList, return_value_policy<manage_new_object>());
    def("feed", raw_function(rawFeed));
    def("uniformInitializer", getUniformInitializer, return_value_policy<manage_new_object>());
    def("fillInitializer", getFillInitializer, return_value_policy<manage_new_object>());

    def("gradientDescentOptim", Number::optimizeByGradientDescent);
    def("momentumOptim", Number::optimizeByMomentum);
    def("RMSPropOptim", Number::optimizeByRMSProp);
    def("adamOptim", Number::optimizeByAdam);
    def("gradientReinit", Number::reinitAllGradient);

    def("gradientDescentOptim", Number::optimizeByGradientDescentAll);
    def("momentumOptim", Number::optimizeByMomentumAll);
    def("RMSPropOptim", Number::optimizeByRMSPropAll);
    def("adamOptim", Number::optimizeByAdamAll);
    def("gradientReinit", Number::reinitAllGradientAll);

    def("Variable", newVariable, return_value_policy<manage_new_object>());
    def("Variable", newSimpleVariable, return_value_policy<manage_new_object>());
    def("Variable", newVariableNumber, return_value_policy<manage_new_object>());
    def("Variable", newVariableNumpy, return_value_policy<manage_new_object>());
    def("Variable", newVariableList, return_value_policy<manage_new_object>());

    def("Variable", newVariableWithGroup, return_value_policy<manage_new_object>());
    def("Variable", newSimpleVariableWithGroup, return_value_policy<manage_new_object>());
    def("Variable", newVariableNumberWithGroup, return_value_policy<manage_new_object>());
    def("Variable", newVariableNumpyWithGroup, return_value_policy<manage_new_object>());
    def("Variable", newVariableListWithGroup, return_value_policy<manage_new_object>());
    def("concatenate", Tensor::concatenate, return_value_policy<manage_new_object>());
    def("stack", Tensor::stack, return_value_policy<manage_new_object>());

    class_<InitializerWrapper>("Initializer", no_init)
        .def("init", &InitializerWrapper::init)
    ;

    void (FeedWrapper::*feed_1)(boost::python::list &values) = &FeedWrapper::feed;
    void (FeedWrapper::*feed_2)(FLOAT value) = &FeedWrapper::feed;
    void (FeedWrapper::*feed_3)(boost::python::numpy::ndarray &a) = &FeedWrapper::feed;
    class_<FeedWrapper, bases<InitializerWrapper>>("Feed", no_init)
        .def("feed", feed_1)
        .def("feed", feed_2)
        .def("feed", feed_3)
    ;

    Tensor *(Tensor::*sum_1)(unsigned int) const = &Tensor::sum;
    Tensor *(Tensor::*sum_2)() const = &Tensor::sum;
    Tensor *(Tensor::*count_1)(unsigned int) const = &Tensor::count;
    Tensor *(Tensor::*count_2)() const = &Tensor::count;
    Tensor *(Tensor::*std_1)(unsigned int) const = &Tensor::std;
    Tensor *(Tensor::*std_2)() const = &Tensor::std;
    Tensor *(Tensor::*max_1)(unsigned int) const = &Tensor::max;
    Tensor *(Tensor::*max_2)() const = &Tensor::max;
    Tensor *(Tensor::*min_1)(unsigned int) const = &Tensor::min;
    Tensor *(Tensor::*min_2)() const = &Tensor::min;
    Tensor *(Tensor::*range_1)(unsigned int) const = &Tensor::range;
    Tensor *(Tensor::*range_2)() const = &Tensor::range;
    Tensor *(Tensor::*mean_1)(unsigned int) const = &Tensor::mean;
    Tensor *(Tensor::*mean_2)() const = &Tensor::mean;
    Tensor *(Tensor::*softmax_1)(unsigned int) const = &Tensor::softmax;
    Tensor *(Tensor::*softmax_2)() const = &Tensor::softmax;
    Tensor *(Tensor::*get_1)(unsigned int) const = &Tensor::get;
    Tensor *(Tensor::*get_2)(Tensor*) const = &Tensor::get;

    class_<Tensor>("Tensor", init<FLOAT>())
        //.def(init<list&, list&>())
        .def(init<np::ndarray&>())
        .def(init<boost::python::list&>())
        .def("__str__", &Tensor::__str__)
        .def("__call__", raw_function(&Tensor::rawEval, 1))
        .def("eval", raw_function(&Tensor::rawEval, 1))
        .def("backpropagation", raw_function(&Tensor::rawPropagation, 1))
        .def("sum", sum_1, return_value_policy<manage_new_object>())
        .def("sum", sum_2, return_value_policy<manage_new_object>())
        .def("count", count_1, return_value_policy<manage_new_object>())
        .def("count", count_2, return_value_policy<manage_new_object>())
        .def("mean", mean_1, return_value_policy<manage_new_object>())
        .def("mean", mean_2, return_value_policy<manage_new_object>())
        .def("softmax", softmax_1, return_value_policy<manage_new_object>())
        .def("softmax", softmax_2, return_value_policy<manage_new_object>())
        .def("std", std_1, return_value_policy<manage_new_object>())
        .def("std", std_2, return_value_policy<manage_new_object>())
        .def("percentile", &Tensor::percentile, return_value_policy<manage_new_object>())
        .def("min", min_1, return_value_policy<manage_new_object>())
        .def("min", min_2, return_value_policy<manage_new_object>())
        .def("max", max_1, return_value_policy<manage_new_object>())
        .def("max", max_2, return_value_policy<manage_new_object>())
        .def("range", range_1, return_value_policy<manage_new_object>())
        .def("range", range_2, return_value_policy<manage_new_object>())
        .def("add", &Tensor::add, return_value_policy<manage_new_object>())
        .def("add", &Tensor::addRaw, return_value_policy<manage_new_object>())
        .def("substract", &Tensor::substract, return_value_policy<manage_new_object>())
        .def("substract", &Tensor::substractRaw, return_value_policy<manage_new_object>())
        .def("pow", &Tensor::pow, return_value_policy<manage_new_object>())
        .def("pow", &Tensor::powRaw, return_value_policy<manage_new_object>())
        .def("multiply", &Tensor::multiply, return_value_policy<manage_new_object>())
        .def("multiply", &Tensor::multiplyRaw, return_value_policy<manage_new_object>())
        .def("divide", &Tensor::divide, return_value_policy<manage_new_object>())
        .def("divide", &Tensor::divideRaw, return_value_policy<manage_new_object>())
        .def("inverse", &Tensor::inverse, return_value_policy<manage_new_object>())
        .def("negative", &Tensor::negative, return_value_policy<manage_new_object>())
        .def("sigmoid", &Tensor::sigmoid, return_value_policy<manage_new_object>())
        .def("cos", &Tensor::cos, return_value_policy<manage_new_object>())
        .def("sin", &Tensor::sin, return_value_policy<manage_new_object>())
        .def("exp", &Tensor::exp, return_value_policy<manage_new_object>())
        .def("log", &Tensor::log, return_value_policy<manage_new_object>())
        .def("matmul", &Tensor::matmul, return_value_policy<manage_new_object>())
        .def("transpose", &Tensor::transposeFromList, return_value_policy<manage_new_object>())
        .def("minor", &Tensor::minorXY, return_value_policy<manage_new_object>())
        .def("determinant", &Tensor::determinant, return_value_policy<manage_new_object>())
        .def("minorMatrix", &Tensor::minorMatrix, return_value_policy<manage_new_object>())
        .def("matinv", &Tensor::matinv, return_value_policy<manage_new_object>())
        .def("shape", &Tensor::shape, return_value_policy<manage_new_object>())
        .def("get", get_1, return_value_policy<manage_new_object>())
        .def("__getitem__", get_1, return_value_policy<manage_new_object>())
        .def("get", get_2, return_value_policy<manage_new_object>())
        .def("__getitem__", get_2, return_value_policy<manage_new_object>())
        .def("gather", &Tensor::gatherFromList, return_value_policy<manage_new_object>())
        .def("gradientChecking", &Tensor::gradientChecking)
        .def("gradientReinit", &Tensor::gradientReinit)
        .def("show", &Tensor::printGradient)

        .add_property("name", &Tensor::getName, &Tensor::setName)
    ;
}

#endif
