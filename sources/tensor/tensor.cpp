#include "tensor/tensor.h"
#include "operation/variable.h"
#include "operation/addition.h"
#include "operation/inverse.h"
#include "operation/sigmoid.h"
#include "operation/multiplication.h"
#include "operation/sum.h"

Tensor::Tensor(std::vector<unsigned int> dims) : dims(dims)
{
    len = calculateLen();
    this->content = new Number*[len];
    for (unsigned int i = 0; i < len; i++) {
        content[i] = NULL;
    }
}

Tensor::Tensor(std::vector<unsigned int> dims, std::vector<float> values) : Tensor(dims)
{
    for (unsigned int i = 0; i < len; i++) {
        setContent(i, new Variable(values[i % values.size()]));
    }
}

Tensor::Tensor(const Tensor *origin, unsigned int idx) : dims(origin->dims)
{
    dims.erase(dims.begin());
    len = calculateLen();
    this->content = new Number*[len];
    for (unsigned int i = 0; i < len; i++) {
        setContent(i, origin->content[len*idx + i]);
    }
}

void Tensor::setContent(unsigned int idx, Number *number) {
    content[idx] = number;
    number->usedBy += 1;
}

void Tensor::unsetContent(unsigned int idx) {
    content[idx]->usedBy -= 1;
    if (content[idx]->usedBy == 0)
        delete content[idx];
}

unsigned int Tensor::calculateLen() {
    unsigned int total = this->dims[0];

    for (unsigned int i = 1; i < this->dims.size(); i++) {
        total *= this->dims[i];
    }
    return total;
}

Tensor::~Tensor() {
    for (unsigned int i = 0; i < len; i++)
        unsetContent(i);
    delete content;
}

Tensor Tensor::shape() {
    return Tensor(
        {(unsigned int)dims.size()},
        std::vector<float>(dims.begin(), dims.end())
    );
}

Tensor Tensor::operator[](unsigned int idx) {
    return get(idx);
}

Tensor Tensor::get(unsigned int idx) const {
    return Tensor(this, idx);
}

unsigned int Tensor::getAbsoluteIndex(std::vector<unsigned int> idx) const {
    if (idx.size() != dims.size()) {
        throw TensorException("Indexs doesn't match tensor dimension");
    }
    unsigned int index = idx[0];
    if (idx[0] >= dims[0])
        throw TensorException("Index out of bounds");
    for (unsigned int i = 1; i < dims.size(); i++) {
        if (idx[i] >= dims[i])
            throw TensorException("Index out of bounds");
        index *= dims[i];
        index += idx[i];
    }
    return index;
}

Number* Tensor::at(std::vector<unsigned int> idx) const {
    unsigned int index = getAbsoluteIndex(idx);
    return content[index];
}

void Tensor::at(std::vector<unsigned int> idx, Number *number) {
    unsigned int index = getAbsoluteIndex(idx);
    setContent(index, number);
}

std::string Tensor::toString(int margin) const {
    
    std::string str = "";
    for (int i =0; i < margin; i++)
        str += " ";
    if (dims.size() == 1) {
        str += "[";
        for (unsigned int i = 0; i < len; i++) {
            if (i == 0)
                str += content[i]->toString();
            else
                str += ", " + content[i]->toString();
        }
        str += "]";
        return str;
    }

    str += "[\n";
    for (unsigned int i = 0; i < dims[0]; i++) {
        str += get(i).toString(margin+1);
        str += "\n";
    }
    for (int i =0; i < margin; i++)
        str += " ";
    str += "]";
    if (margin == 0)
        str += NL;
    return str;
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    std::string str = tensor.toString();
    os << str;
    return os;
}

bool Tensor::equals(Tensor &tensor) {
    if (sameShape(tensor) == false)
        return false;
    for (unsigned int i = 0; i < len; i++) {
        if (content[i]->equals(*tensor.content[i]) == false)
            return false;
    }
    return true;
}

bool Tensor::sameShape(Tensor &tensor) {
    if (dims.size() != tensor.dims.size())
        return false;
    for (unsigned int i = 0; i < dims.size(); i++) {
        if (dims[i] != tensor.dims[i])
            return false;
    }
    return true;
}

Tensor Tensor::add(Number &number) {
    Tensor result(dims);
    for (unsigned int i = 0; i < len; i++) {
        result.setContent(i, new Addition(content[i], &number));
    }
    return result;
}

Tensor Tensor::add(Tensor &tensor) {
    if (sameShape(tensor) == false)
        throw TensorException("can not add tensor of different shapes");
    Tensor result(dims);
    for (unsigned int i = 0; i < len; i++) {
        result.setContent(i, new Addition(content[i], tensor.content[i]));
    }
    return result;
}

Tensor Tensor::multiply(Number &number) {
    Tensor result(dims);
    for (unsigned int i = 0; i < len; i++) {
        result.setContent(i, new Multiplication(content[i], &number));
    }
    return result;
}

Tensor Tensor::multiply(Tensor &tensor) {
    if (sameShape(tensor) == false)
        throw TensorException("can not add tensor of different shapes");
    Tensor result(dims);
    for (unsigned int i = 0; i < len; i++) {
        result.setContent(i, new Multiplication(content[i], tensor.content[i]));
    }
    return result;
}

Tensor Tensor::inverse() {
    Tensor result(dims);
    for (unsigned int i = 0; i < len; i++) {
        result.setContent(i, new Inverse(content[i]));
    }
    return result;
}

Tensor Tensor::sigmoid() {
    Tensor result(dims);
    for (unsigned int i = 0; i < len; i++) {
        result.setContent(i, new Sigmoid(content[i]));
    }
    return result;
}

Tensor Tensor::matmul(Tensor &tensor) {
    if (dims.size() != 2 || tensor.dims.size() != 2)
        throw TensorException("Matrix multiplication can only be done with 2 dimensional matrix");
    if (dims[1] != tensor.dims[0])
        throw TensorException("Matrix multiplication dimensions doesn't match");
    Tensor result({dims[0], tensor.dims[1]});
    for (unsigned int i = 0; i < dims[0]; i++) {
        for (unsigned int j = 0; j < tensor.dims[1]; j++) {
            std::vector<Number*> row;
            for (unsigned int k = 0; k < dims[1]; k++) {
                row.push_back(new Multiplication(at({i, k}), tensor.at({k, j})));
            }
            result.at({i,j}, new Sum(row));
        }
    }
    return result;
}

