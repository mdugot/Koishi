#include "tensor/tensor.h"
#include "operation/variable.h"

Tensor::Tensor(std::vector<unsigned int> dims, float value) : Tensor()
{
    int len = dims[0];
    if (dims.size() > 1) {
        dims.erase(dims.begin());
        for (int i = 0; i < len; i++) {
            units.push_back(new Tensor(dims, value));
        }
    } else {
        for (int i = 0; i < len; i++) {
            units.push_back(new Variable(value));
        }
    }
}

Tensor::Tensor(std::vector<float> values) : Tensor()
{
    for (float v : values) {
        units.push_back(new Variable(v));
    }
}

Tensor::Tensor()
{
    isNumber = false;
}

Tensor::~Tensor() {
    for (TensorInterface *t : units) {
        delete t;
    }
}

int Tensor::len() {
    return units.size();
}

Tensor Tensor::shape() {
    TensorInterface *tmp = this;
    std::vector<float> values;
    while (tmp->isNumber == false) {
        values.push_back((float)tmp->len());
        tmp = &((*tmp)[0]);
    }
    return Tensor(values);
}

TensorInterface& Tensor::operator[](unsigned int idx) {
    if (idx >= units.size())
        throw TensorIndexException("index out of bound when select tensor element", idx, units.size());
    return *units[idx];
}

std::string Tensor::toString(int margin) {
    std::string str = "";
    for (int i = 0; i<margin; i++)
        str += " ";
    str += "[";
    bool vector = false;
    for (unsigned int n = 0; n < units.size(); n++) {
        if (units[n]->isNumber == false) {
            str += "\n";
//            for (int i = 0; i<=margin; i++)
//                str += " ";
            str += units[n]->toString(margin+1);
        } else {
            vector = true;
            str += units[n]->toString(0);
        }
        if (n < units.size() -1) {
            str += ", ";
        }
    }
    if (vector)
        str += "]";
    else {
        str += "\n";
        for (int i = 0; i<margin; i++)
            str += " ";
        str += "]";
    }
    return str;
}

bool Tensor::equals(TensorInterface &tensor) {
    return tensor.equals(*this);
}

bool Tensor::equals(Tensor &tensor) {
    if (len() != tensor.len())
        return false;
    for (int i = 0; i < len(); i++) {
        if ((*this)[i] != tensor[i])
            return false;
    }
    return true;
}

bool Tensor::equals(Number &number) {
    (void)number;
    return false;
}
