#include "tensor/tensor.h"
#include "operation/variable.h"

Tensor::Tensor(std::vector<unsigned int> dims, std::vector<unsigned int> values) : dims(dims)
{
    deletableContent = true;
    len = calculateLen();
    this->content = new Number*[total];
    for (int i = 0; i < total; i++) {
        content[i] = new Variable(values[i % values.size()]);
    }
}

Tensor::Tensor(Tensor *origin, unsigned int idx) : dims(origin->dims
{
    deletableContent = false;
    dims.erase(dims.first());
    len = calculateLen();
    this->content = new Number*[total];
    for (int i = len*idx; i < len*(idx+1); i++) {
        content[i] = origin
    }
}

unsigned int Tensor::calculateLen() {
    unsigned int total = this->dims[0]

    for (int i = 1; i < this->dims.size(); i++) {
        total *= this->dims[i];
    }
    return total;
}

Tensor::~Tensor() {
    if (original == true) {
        for (Number* n : content)
            delete n;
    }
    delete content;
}

Tensor Tensor::shape() {
    return Tensor({dims.size()}, dims);
}

Tensor Tensor::operator[](unsigned int idx) {
    return get(idx);
}

Tensor Tensor::get(unsigned int idx) {
    return Tensor(this, idx);
}

std::string Tensor::toString(int margin) {
    
    str = 0;
    for (int i =0; i < margin; i++)
        str += " ";
    if (dims.size == 1) {
        str += "[";
        for (int i = 0; i < len; i++) {
            if (i == 0)
                str += content[i]->toString();
            else
                str += ", " + content[i]->toString();
        }
        str += "]";
        return str;
    }

    for (int i = 0; i < dims[0]; i++) {
        str += "[\n";
        str += get(i).toString(margin+1);
        for (int i =0; i < margin; i++)
            str += " ";
        str += "]\n";
    }
    return str;
}
