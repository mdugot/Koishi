#include "tensor/tensors.h"

bool incrementVectorIdx(std::vector<unsigned int> &dims, std::vector<unsigned int> &idx)
{
    for (int I = idx.size() - 1; I >= 0; I--)
    {
        unsigned int i = (unsigned int)I;
        if (idx[i] < dims[i]-1) {
            idx[i] = idx[i] + 1;
            return true;
        } else {
            if (I == 0)
                return false;
            idx[i] = 0;
        }
    }
    return false;
}

Tensors::Tensors(const Tensor *from, unsigned int splitAxis)
{
    std::vector<unsigned int> dims = from->getDims();
    if (splitAxis >= dims.size())
        throw TensorException("split axis greater than origin dimension", from);
    std::vector<unsigned int> idx(splitAxis+1, 0);
    do {
        Tensor *tensor = from->gather(idx);
        this->content.push_back(tensor);
    } while (incrementVectorIdx(dims, idx));
}

Tensors::Tensors()
{
}

Tensors::~Tensors()
{
    for (unsigned int i = 0; i < content.size(); i++) {
        delete content[i];
    }
}


std::string Tensors::toString()
{
    std::string str = "";
    for (unsigned int i = 0; i < content.size(); i++) {
        str += content[i]->toString();
        str += "\n";
    }
    return str;
}

void Tensors::append(Tensor *tensor) {
    if (content.size() > 0) {
        if (content[0]->sameShape(*tensor) == false)
            throw TensorException("can not append tensor of different shapes", tensor);
    }
    content.push_back(tensor);
}

Tensor* Tensors::get(unsigned int i) const {
    if (i >= content.size())
        throw TensorException("index out of bound");
    return content[i];
}

Tensor* Tensors::merge(std::vector<unsigned int> dims) {
    unsigned int total = 1;
    for (unsigned int i = 0; i < dims.size(); i++) {
        total *= dims[i];
    }
    if (total != content.size())
        throw TensorException("merge dimensions do not match total number of elements");
    auto contentDims = content[0]->getDims();
    for (unsigned int i = 0; i < contentDims.size(); i++) {
        dims.push_back(contentDims[i]);
    }
    return new Tensor(this, dims);
}
