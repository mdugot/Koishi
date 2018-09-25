#include "tensor/tensor.h"
#include "operation/variable.h"
#include "operation/constant.h"
#include "operation/addition.h"
#include "operation/inverse.h"
#include "operation/sigmoid.h"
#include "operation/multiplication.h"
#include "operation/sum.h"

unsigned int Tensor::count = 0;

Tensor::Tensor(std::vector<unsigned int> dims) : dims(dims)
{
    count += 1;
    static unsigned int c = 0;
    c+=1;
    name = "tensor" + std::to_string(c);
    len = calculateLen();
    this->content = new Number*[len];
    for (unsigned int i = 0; i < len; i++) {
        content[i] = NULL;
    }
}

Tensor::Tensor(std::vector<unsigned int> dims, std::vector<FLOAT> values) : Tensor(dims)
{
    for (unsigned int i = 0; i < len; i++) {
        setContent(i, new Constant(values[i % values.size()]));
    }
}

Tensor::Tensor(std::vector<unsigned int> dims, std::string group, Initializer &initializer) : Tensor(dims)
{
    for (unsigned int i = 0; i < len; i++) {
        std::string name = this->name + "_" + std::to_string(i);
        setContent(i, new Variable(group, name, initializer));
    }
}

Tensor::Tensor(const Tensor *origin, unsigned int idx) : dims(origin->dims)
{
    count += 1;
    name = origin->name + "[" + std::to_string(idx) + "]";
    dims.erase(dims.begin());
    len = calculateLen();
    this->content = new Number*[len];
    for (unsigned int i = 0; i < len; i++) {
        setContent(i, origin->content[len*idx + i]);
    }
}

Tensor::~Tensor() {
    count -= 1;
    for (unsigned int i = 0; i < len; i++)
        unsetContent(i);
    delete content;
}

void Tensor::setContent(unsigned int idx, Number *number) {
    content[idx] = number;
    number->usedBy += 1;
}

void Tensor::unsetContent(unsigned int idx) {
    content[idx]->unset();
}

unsigned int Tensor::calculateLen() {
    unsigned int total = this->dims[0];

    for (unsigned int i = 1; i < this->dims.size(); i++) {
        total *= this->dims[i];
    }
    return total;
}

void Tensor::calculateGradient() {
    for (unsigned int i = 0; i < len; i++) {
        content[i]->calculateGradient();
    }
}

Tensor Tensor::shape() {
    Tensor result = Tensor(
        {(unsigned int)dims.size()},
        std::vector<FLOAT>(dims.begin(), dims.end())
    );
    result.name = name + "_shape";
    return result;
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

std::string Tensor::header() const {
    std::string shapeStr = " (shape="; 
    for (unsigned int i = 0; i < dims.size(); i++) {
        if (i > 0)
            shapeStr += ",";
        shapeStr += std::to_string(dims[i]);
    }
    shapeStr += ")";
    return name + shapeStr;
}

std::string Tensor::toString(bool printGradient, int margin) const {
    
    std::string str = "";
    if (margin == 0) {
        str += header() + " : " + NL;
    }
    for (int i =0; i < margin; i++)
        str += " ";
    if (dims.size() == 1) {
        str += "[";
        for (unsigned int i = 0; i < len; i++) {
            if (i == 0)
                str += content[i]->toString(printGradient);
            else
                str += ", " + content[i]->toString(printGradient);
        }
        str += "]";
        return str;
    }

    str += "[\n";
    for (unsigned int i = 0; i < dims[0]; i++) {
        str += get(i).toString(printGradient, margin+1);
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
    static unsigned int c = 0;
    Tensor result(dims);
    for (unsigned int i = 0; i < len; i++) {
        result.setContent(i, new Addition(content[i], &number));
    }
    c+=1;
    result.name = "scallarAdd" + std::to_string(c);
    return result;
}

Tensor Tensor::add(Tensor &tensor) {
    static unsigned int c = 0;
    if (sameShape(tensor) == false)
        throw TensorException("can not add tensor of different shapes");
    Tensor result(dims);
    for (unsigned int i = 0; i < len; i++) {
        result.setContent(i, new Addition(content[i], tensor.content[i]));
    }
    c+=1;
    result.name = "add" + std::to_string(c);
    return result;
}

Tensor Tensor::multiply(Number &number) {
    static unsigned int c = 0;
    Tensor result(dims);
    for (unsigned int i = 0; i < len; i++) {
        result.setContent(i, new Multiplication(content[i], &number));
    }
    c+=1;
    result.name = "scallarMultiply" + std::to_string(c);
    return result;
}

Tensor Tensor::multiply(Tensor &tensor) {
    static unsigned int c = 0;
    if (sameShape(tensor) == false)
        throw TensorException("can not add tensor of different shapes");
    Tensor result(dims);
    for (unsigned int i = 0; i < len; i++) {
        result.setContent(i, new Multiplication(content[i], tensor.content[i]));
    }
    c+=1;
    result.name = "multiply" + std::to_string(c);
    return result;
}

Tensor Tensor::inverse() {
    static unsigned int c = 0;
    Tensor result(dims);
    for (unsigned int i = 0; i < len; i++) {
        result.setContent(i, new Inverse(content[i]));
    }
    c+=1;
    result.name = "inverse" + std::to_string(c);
    return result;
}

Tensor Tensor::sigmoid() {
    static unsigned int c = 0;
    Tensor result(dims);
    for (unsigned int i = 0; i < len; i++) {
        result.setContent(i, new Sigmoid(content[i]));
    }
    c+=1;
    result.name = "sigmoid" + std::to_string(c);
    return result;
}

Tensor Tensor::matmul(Tensor &tensor) {
    static unsigned int c = 0;

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
    c+=1;
    result.name = "matmul" + std::to_string(c);
    return result;
}

Sum Tensor::sum() {
    std::vector<Number*> vector;
    for (unsigned int i = 0; i < len; i++) {
        vector.push_back(content[i]);
    }
    return Sum(vector);
}
