#ifdef PYTHON_WRAPPER
#include "tensor/tensor.h"

char const* greet()
{
   return "KOISHI (a math library for machine-learning)";
}

BOOST_PYTHON_MODULE(koishi)
{
    def("greet", greet);

    def("uniformInitializer", getUniformInitializer, return_value_policy<manage_new_object>());
    def("feedInitializer", getFeedInitializer, return_value_policy<manage_new_object>());

    def("gradientDescent", Number::gradientDescent);
    def("gradientReinit", Number::reinitAllGradient);

    class_<InitializerWrapper>("Initializer", no_init)
        .def("init", &InitializerWrapper::init)
    ;
    class_<FeedWrapper, bases<InitializerWrapper>>("Feed", no_init)
        .def("feed", &FeedWrapper::feed)
    ;

    class_<Tensor>("Tensor", init<FLOAT>())
        .def(init<list&, list&>())
        .def(init<list&, std::string, InitializerWrapper&>())
        .def(init<std::string, InitializerWrapper&>())
        .def("__str__", &Tensor::__str__)
        .def("sum", &Tensor::sum, return_value_policy<manage_new_object>())
        .def("add", &Tensor::add, return_value_policy<manage_new_object>())
        .def("pow", &Tensor::pow, return_value_policy<manage_new_object>())
        .def("multiply", &Tensor::multiply, return_value_policy<manage_new_object>())
        .def("inverse", &Tensor::inverse, return_value_policy<manage_new_object>())
        .def("sigmoid", &Tensor::sigmoid, return_value_policy<manage_new_object>())
        .def("matmul", &Tensor::matmul, return_value_policy<manage_new_object>())
        .def("shape", &Tensor::shape, return_value_policy<manage_new_object>())
        .def("__getitem__", &Tensor::get, return_value_policy<manage_new_object>())
        .def("gradientUpdate", &Tensor::gradientUpdate)
        .def("gradientChecking", &Tensor::gradientChecking)
        .def("show", &Tensor::printGradient)

        .add_property("name", &Tensor::getName, &Tensor::setName)
    ;
}

#endif
