#include "tensor/tensorException.h"

TensorException::TensorException(std::string message) : message(message)
{
}

const char* TensorException::what() const throw() {
    return message.c_str();
}

