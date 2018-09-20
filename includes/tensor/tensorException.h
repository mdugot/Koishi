#ifndef TENSOREXCEPTION_H
#define TENSOREXCEPTION_H
#include "utils.h"

class TensorException : public std::exception {

    private:
        std::string message;
    public:
        TensorException(std::string message);
        virtual const char* what() const throw();
    
};
#endif

