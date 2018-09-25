#ifndef TENSOR_H
#define TENSOR_H
#include "utils.h"
#include "tensor/tensorException.h"
#include "operation/number.h"
#include "initializer/initializer.h"

class Sum;

class Tensor {

    private:
        Tensor(std::vector<unsigned int> dims);
        std::vector<unsigned int> dims;
        unsigned int len;
        Number** content;
        std::string name;
        

    public:
        static unsigned int count;

        Tensor(std::vector<unsigned int> dims, std::vector<float> values);
        Tensor(std::vector<unsigned int> dims, std::string group, Initializer &initializer);
        Tensor(const Tensor *origin, unsigned int idx);
        ~Tensor();
        inline void setName(std::string str) {name = str;}
        inline std::string getName() {return name;}
        void setContent(unsigned int idx, Number *number);
        void unsetContent(unsigned int idx);
        Tensor operator[](unsigned int idx);
        Tensor get(unsigned int idx) const;
        unsigned int calculateLen();
        void calculateGradient();
        Tensor shape();
        std::string toString(bool printGradient = false, int margin = 0) const;
        std::string header() const;
        bool equals(Tensor &tensor);
        bool sameShape(Tensor &tensor);
        Tensor add(Tensor &tensor);
        Tensor add(Number &number);
        Tensor multiply(Tensor &tensor);
        Tensor multiply(Number &number);
        Tensor inverse();
        Tensor sigmoid();
        Sum sum();
        unsigned int getAbsoluteIndex(std::vector<unsigned int>idx) const;
        Number* at(std::vector<unsigned int>idx) const;
        void at(std::vector<unsigned int>idx, Number* number);
        Tensor matmul(Tensor &tensor);
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

#endif
