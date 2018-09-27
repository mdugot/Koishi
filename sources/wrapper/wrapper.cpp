#ifdef PYTHON_WRAPPER
#include "tensor/tensor.h"

char const* greet()
{
   return "hello, world";
}

BOOST_PYTHON_MODULE(koishi)
{
    def("greet", greet);

    class_<InitializerWrapper>("Initializer", init<std::string, FLOAT, FLOAT>())
        .def("init", &InitializerWrapper::init)
    ;

    class_<Tensor>("Tensor", init<FLOAT>())
        .def(init<list&, list&>())
        .def(init<list&, std::string, InitializerWrapper&>())
        .def(init<std::string, InitializerWrapper&>())
        .def("__str__", &Tensor::__str__)
        .def("add", &Tensor::add, return_value_policy<manage_new_object>())
        .def("multiply", &Tensor::multiply, return_value_policy<manage_new_object>())
        .def("inverse", &Tensor::inverse, return_value_policy<manage_new_object>())
        .def("sigmoid", &Tensor::sigmoid, return_value_policy<manage_new_object>())
        .def("matmul", &Tensor::matmul, return_value_policy<manage_new_object>())
        .def("sum", &Tensor::sum, return_value_policy<manage_new_object>())
        .def("shape", &Tensor::shape, return_value_policy<manage_new_object>())
        .def("__getitem__", &Tensor::get, return_value_policy<manage_new_object>())
    ;
}

#endif
