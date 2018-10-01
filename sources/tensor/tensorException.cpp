#include "tensor/tensorException.h"
#include "tensor/tensor.h"

TensorException::TensorException(std::string message) : message(message)
{
}

TensorException::TensorException(std::string message, const Tensor *t1) : TensorException(message)
{
        this->message += " : " + t1->header();
}

TensorException::TensorException(std::string message, const Tensor *t1, const Tensor *t2) : TensorException(message, t1)
{
        this->message += " and " + t2->header();
}

const char* TensorException::what() const throw() {
    return message.c_str();
}

