#ifndef TENSOR_H
#define TENSOR_H
#include "utils.h"
#include "tensor/tensorIndexException.h"
#include "operation/number.h"

class Tensor {

    private:
        bool deletableContent;
        std::vector<unsigned int> dims;
        unsigned int len;
        Number** content;
        

    public:
        Tensor(std::vector<unsigned int> dims, std::vector<unsigned int> values = {0});
        Tensor(Tensor *origin, unsigned int idx);
        ~Tensor();
        Tensor operator [](unsigned int idx);
        Tensor shape();
        std::string toString();
//        bool equals(Tensor &tensor);
        
    friend class Tensor;
};

#endif
