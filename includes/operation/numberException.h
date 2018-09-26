#ifndef NUMBEREXCEPTION_H
#define NUMBEREXCEPTION_H
#include "utils.h"

class NumberException : public std::exception {

    private:
        std::string message;
    public:
        NumberException(std::string message);
        virtual const char* what() const throw();
    
};
#endif

