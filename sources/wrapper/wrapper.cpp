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

    def("uniformInitializer", getUniformInitializer, return_value_policy<manage_new_object>());
    def("feedInitializer", getFeedInitializer, return_value_policy<manage_new_object>());
    def("feedInitializer", getFeedInitializerFromList, return_value_policy<manage_new_object>());
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

    class_<InitializerWrapper>("Initializer", no_init)
        .def("init", &InitializerWrapper::init)
    ;
    class_<FeedWrapper, bases<InitializerWrapper>>("Feed", no_init)
        .def("feed", &FeedWrapper::feed)
        .def("feed", &FeedWrapper::feedSimple)
        .def("feed", &FeedWrapper::feedNumpy)
    ;

    Tensor *(Tensor::*mean_1)(unsigned int) const = &Tensor::mean;
    Tensor *(Tensor::*mean_2)() const = &Tensor::mean;

    class_<Tensor>("Tensor", init<FLOAT>())
        //.def(init<list&, list&>())
        .def(init<np::ndarray&>())
        .def(init<boost::python::list&>())
        .def("__str__", &Tensor::__str__)
        .def("eval", &Tensor::evalForPython)
        .def("sum", &Tensor::sum, return_value_policy<manage_new_object>())
        .def("count", &Tensor::count, return_value_policy<manage_new_object>())
        .def("mean", mean_1, return_value_policy<manage_new_object>())
        .def("mean", mean_2, return_value_policy<manage_new_object>())
        .def("std", &Tensor::std, return_value_policy<manage_new_object>())
        .def("percentile", &Tensor::percentile, return_value_policy<manage_new_object>())
        .def("min", &Tensor::min, return_value_policy<manage_new_object>())
        .def("max", &Tensor::max, return_value_policy<manage_new_object>())
        .def("range", &Tensor::range, return_value_policy<manage_new_object>())
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
        .def("exp", &Tensor::exp, return_value_policy<manage_new_object>())
        .def("log", &Tensor::log, return_value_policy<manage_new_object>())
        .def("matmul", &Tensor::matmul, return_value_policy<manage_new_object>())
        .def("transpose", &Tensor::transposeFromList, return_value_policy<manage_new_object>())
        .def("minor", &Tensor::minorXY, return_value_policy<manage_new_object>())
        .def("determinant", &Tensor::determinant, return_value_policy<manage_new_object>())
        .def("minorMatrix", &Tensor::minorMatrix, return_value_policy<manage_new_object>())
        .def("matinv", &Tensor::matinv, return_value_policy<manage_new_object>())
        .def("shape", &Tensor::shape, return_value_policy<manage_new_object>())
        .def("__getitem__", &Tensor::get, return_value_policy<manage_new_object>())
        .def("gather", &Tensor::gatherFromList, return_value_policy<manage_new_object>())
        .def("backpropagation", &Tensor::gradientUpdate)
        .def("gradientChecking", &Tensor::gradientChecking)
        .def("gradientReinit", &Tensor::gradientReinit)
        .def("show", &Tensor::printGradient)

        .add_property("name", &Tensor::getName, &Tensor::setName)
    ;
}

#endif
