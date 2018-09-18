#include "tensor/tensor.h"
#include "operation/variable.h"

Tensor::Tensor(std::vector<unsigned int> dims, float value)
{
    isNumber = false;
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

TensorInterface* Tensor::operator[](unsigned int idx) {
    if (idx >= units.size())
        throw TensorIndexException("index out of bound when select tensor element", idx, units.size());
    return units[idx];
}

std::string Tensor::toString(int margin) {
    std::string str = "";
    for (int i = 0; i<margin; i++)
        str += " ";
    str += "[";
    for (unsigned int n = 0; n < units.size(); n++) {
        if (units[n]->isNumber == false) {
            str += "\n";
            for (int i = 0; i<margin; i++)
                str += " ";
            str += units[n]->toString(margin+1);
        } else {
            str += units[n]->toString(0);
        }
        if (n < units.size() -1) {
            str += ", ";
        }
    }
    str += "]";
    return str;
}
