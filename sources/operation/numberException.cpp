#include "operation/numberException.h"

NumberException::NumberException(std::string message) : message(message)
{
}

const char* NumberException::what() const throw() {
    return message.c_str();
}

