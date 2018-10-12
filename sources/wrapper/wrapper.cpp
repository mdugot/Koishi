#ifdef PYTHON_WRAPPER
#include "tensor/tensor.h"
#include "operation/variable.h"

char const* greet()
{
   return "KOISHI (a math library for machine-learning)";
}

BOOST_PYTHON_MODULE(koishi)
{
    Py_Initialize();
    np::initialize();

    def("greet", greet);
    

    def("initializeAll", &Initializer::initializeAll);
    def("save", &Variable::save);
    def("saveAll", &Variable::saveAll);
    def("load", &Variable::load);

    def("uniformInitializer", getUniformInitializer, return_value_policy<manage_new_object>());
    def("feedInitializer", getFeedInitializer, return_value_policy<manage_new_object>());
    def("fillInitializer", getFillInitializer, return_value_policy<manage_new_object>());

    def("gradientDescentOptim", Number::optimizeByGradientDescent);
    def("momentumOptim", Number::optimizeByMomentum);
    def("RMSPropOptim", Number::optimizeByRMSProp);
    def("adamOptim", Number::optimizeByAdam);
    def("gradientReinit", Number::reinitAllGradient);

    class_<InitializerWrapper>("Initializer", no_init)
        .def("init", &InitializerWrapper::init)
    ;
    class_<FeedWrapper, bases<InitializerWrapper>>("Feed", no_init)
        .def("feed", &FeedWrapper::feed)
        .def("feed", &FeedWrapper::feedSimple)
        .def("feed", &FeedWrapper::feedNumpy)
    ;

    class_<Tensor>("Tensor", init<FLOAT>())
        .def(init<list&, list&>())
        .def(init<np::ndarray&>())
        .def(init<list&, std::string, InitializerWrapper&>())
        .def(init<std::string, InitializerWrapper&>())
        .def("__str__", &Tensor::__str__)
        .def("eval", &Tensor::evalForPython)
        .def("sum", &Tensor::sum, return_value_policy<manage_new_object>())
        .def("count", &Tensor::count, return_value_policy<manage_new_object>())
        .def("mean", &Tensor::mean, return_value_policy<manage_new_object>())
        .def("std", &Tensor::std, return_value_policy<manage_new_object>())
        .def("percentile", &Tensor::percentile, return_value_policy<manage_new_object>())
        .def("min", &Tensor::min, return_value_policy<manage_new_object>())
        .def("max", &Tensor::max, return_value_policy<manage_new_object>())
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
        .def("sigmoid", &Tensor::sigmoid, return_value_policy<manage_new_object>())
        .def("matmul", &Tensor::matmul, return_value_policy<manage_new_object>())
        .def("shape", &Tensor::shape, return_value_policy<manage_new_object>())
        .def("__getitem__", &Tensor::get, return_value_policy<manage_new_object>())
        .def("gradientUpdate", &Tensor::gradientUpdate)
        .def("gradientChecking", &Tensor::gradientChecking)
        .def("gradientReinit", &Tensor::gradientReinit)
        .def("show", &Tensor::printGradient)

        .add_property("name", &Tensor::getName, &Tensor::setName)
    ;
}

#endif
