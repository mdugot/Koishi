#ifndef TENSOR_H
#define TENSOR_H
#include "utils.h"
#include "tensor/tensorInterface.h"
#include "tensor/tensorIndexException.h"

class Tensor : public TensorInterface {

    private:
        std::vector<TensorInterface*> units;

    public:
        Tensor(std::vector<unsigned int> dims, float value = 0);
        TensorInterface* operator [](unsigned int idx);
        std::string toString(int margin = 0);
    
};
#endif
