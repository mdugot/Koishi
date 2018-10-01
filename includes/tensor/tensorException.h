#ifndef TENSOREXCEPTION_H
#define TENSOREXCEPTION_H
#include "utils.h"

class Tensor;
class TensorException : public std::exception {

    private:
        std::string message;
    public:
        TensorException(std::string message);
        TensorException(std::string message, const Tensor *t1);
        TensorException(std::string message, const Tensor *t1, const Tensor *t2);
        virtual const char* what() const throw();
    
};
#endif

