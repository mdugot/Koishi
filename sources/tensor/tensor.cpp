#include "tensor/tensor.h"
#include "operation/variable.h"
#include "operation/addition.h"
#include "operation/inverse.h"
#include "operation/sigmoid.h"
#include "operation/multiplication.h"

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
