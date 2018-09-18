#include "tensor/tensorIndexException.h"

TensorIndexException::TensorIndexException(std::string message, int index, int max, int min) :message(message), index(index), max(max), min(min)
{
}

const char* TensorIndexException::what() const throw() {
    std::string str;
    if (max >= min)
        str = message + " (index = " + std::to_string(index) + ") : index must be comprise betwwen " + std::to_string(min) + " and " + std::to_string(max);
    else
        str = message + " (index = " + std::to_string(index) + ")";
    return str.c_str();
}
