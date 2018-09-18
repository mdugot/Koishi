#ifndef TENSORINDEXEXCEPTION_H
#define TENSORINDEXEXCEPTION_H
#include "utils.h"

class TensorIndexException : public std::exception {

    private:
        std::string message;
        int index;
        int max;
        int min;
    public:
        TensorIndexException(std::string message, int index, int max = -1, int min = 0);
        virtual const char* what() const throw();
    
};
#endif

