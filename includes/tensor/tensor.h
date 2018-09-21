#ifndef TENSOR_H
#define TENSOR_H
#include "utils.h"
#include "tensor/tensorException.h"
#include "operation/number.h"

class Tensor {

    private:
        Tensor(std::vector<unsigned int> dims);
        std::vector<unsigned int> dims;
        unsigned int len;
        Number** content;
        

    public:
        Tensor(std::vector<unsigned int> dims, std::vector<float> values);
        Tensor(const Tensor *origin, unsigned int idx);
        ~Tensor();
        void setContent(unsigned int idx, Number *number);
        void unsetContent(unsigned int idx);
        Tensor operator[](unsigned int idx);
        Tensor get(unsigned int idx) const;
        unsigned int calculateLen();
        Tensor shape();
        std::string toString(int margin = 0) const;
        bool equals(Tensor &tensor);
        bool sameShape(Tensor &tensor);
        Tensor add(Tensor &tensor);
        Tensor add(Number &number);
        Tensor multiply(Tensor &tensor);
        Tensor multiply(Number &number);
        Tensor inverse();
        Tensor sigmoid();
        unsigned int getAbsoluteIndex(std::vector<unsigned int>idx) const;
        Number* at(std::vector<unsigned int>idx) const;
        void at(std::vector<unsigned int>idx, Number* number);
        Tensor matmul(Tensor &tensor);
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

#endif
