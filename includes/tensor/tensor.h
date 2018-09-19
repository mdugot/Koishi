#ifndef TENSOR_H
#define TENSOR_H
#include "utils.h"
#include "tensor/tensorInterface.h"
#include "tensor/tensorIndexException.h"

class Tensor : public TensorInterface {

    private:
        std::vector<TensorInterface*> units;

    public:
        Tensor(std::vector<unsigned int> dims, float value);
        Tensor(std::vector<float> values);
        Tensor();
        ~Tensor();
        TensorInterface& operator [](unsigned int idx);
        int len();
        Tensor shape();
        std::string toString(int margin = 0);
        inline Tensor* self() {return this;}
        virtual bool equals(Tensor &tensor);
        virtual bool equals(Number &number);
        virtual bool equals(TensorInterface &tensor);
        
    
};
#endif
