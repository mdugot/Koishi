#include "tensor/tensor.h"
#include "operation/get.h"
#include "operation/variable.h"
#include "operation/constant.h"
#include "operation/addition.h"
#include "operation/inverse.h"
#include "operation/sigmoid.h"
#include "operation/cosine.h"
#include "operation/exp.h"
#include "operation/log.h"
#include "operation/multiplication.h"
#include "operation/sum.h"
#include "operation/pow.h"
#include "operation/count.h"
#include "operation/percentile.h"

unsigned int Tensor::counter = 0;

#ifdef PYTHON_WRAPPER
#include "wrapper/wrapperTools.h"

    Tensor::Tensor(np::ndarray &a)
    : Tensor(
        getNumpyShape(a),
        numpyToVector(a)
    ){}

    Tensor::Tensor(list &l)
    : Tensor(
        getListShape(l),
        listToVector_deep<FLOAT>(l)
    ){}

    Tensor::Tensor(std::string group, np::ndarray &a)
    : Tensor(
        getNumpyShape(a),
        group,
        (Initializer*)NULL
    ){
        std::vector<FLOAT> values = numpyToVector(a);
        for (unsigned int i = 0; i < len; i++) {
            ((Constant*)content[i])->setValue(values[i]);
        }
        if (len % values.size() != 0)
            throw TensorException("values of initialization must be a divisor of the len of tensor", this);
    }

    Tensor::Tensor(std::string group, boost::python::list &list)
    : Tensor(
        getListShape(list),
        group,
        (Initializer*)NULL
    ){
        std::vector<FLOAT> values = listToVector_deep<FLOAT>(list);
        for (unsigned int i = 0; i < len; i++) {
            ((Constant*)content[i])->setValue(values[i]);
        }
        if (len % values.size() != 0)
            throw TensorException("values of initialization must be a divisor of the len of tensor", this);
    }

    Tensor::Tensor(std::string group, FLOAT value)
    : Tensor(
        group,
        (Initializer*)NULL
    ){
        ((Constant*)content[0])->setValue(value);
    }

    Tensor::Tensor(list& dims, std::string group, InitializerWrapper &wrap)
    : Tensor(
        listToVector<unsigned int>(dims),
        group,
        wrap.initializer
    ){}

    Tensor::Tensor(std::string group, InitializerWrapper &wrap)
    : Tensor(
        wrap.initializer->getDims(),
        group,
        wrap.initializer
    ){}

    Tensor *Tensor::transposeFromList(list& permutations) {
        return transpose(listToVector<unsigned int>(permutations));
    }

    Tensor *Tensor::gatherFromList(list& idx) {
        return gather(listToVector<unsigned int>(idx));
    }
#endif

Tensor::Tensor(std::vector<unsigned int> dims) : dims(dims)
{
    counter += 1;
    static unsigned int c = 0;
    c+=1;
    name = "tensor" + std::to_string(c);
    len = calculateLen();
    this->content = new Number*[len];
    for (unsigned int i = 0; i < len; i++) {
        content[i] = NULL;
    }
}

Tensor::Tensor(const Tensor *origin, unsigned int idx) : dims(origin->dims)
{
    counter += 1;
    name = origin->name + "[" + std::to_string(idx) + "]";
    dims.erase(dims.begin());
    len = calculateLen();
    this->content = new Number*[len];
    for (unsigned int i = 0; i < len; i++) {
        content[i] = NULL;
        setContent(i, origin->content[len*idx + i]);
    }
}

Tensor::Tensor(const Tensor *origin) : dims(origin->dims)
{
    counter += 1;
    name = origin->name + "_copy";
    len = calculateLen();
    this->content = new Number*[len];
    for (unsigned int i = 0; i < len; i++) {
        content[i] = NULL;
        setContent(i, origin->content[i]);
    }
}

Tensor::Tensor(const std::vector<Tensor*> tensors) : dims()
{
    counter += 1;
    name = "merge(";
    dims.push_back(tensors.size());
    if (tensors[0]->dims.size() > 0) {
        dims.insert(dims.end(), tensors[0]->dims.begin(), tensors[0]->dims.end());
    }

    len = calculateLen();
    this->content = new Number*[len];
    int c_i = 0;
    for (int idx = 0; idx < tensors.size(); idx++) {
        if (tensors[idx]->dims != tensors[0]->dims)
            throw TensorException("try to merge tensors of different shape", this);
        name += tensors[idx]->name + ",";
        for (unsigned int i = 0; i < tensors[idx]->len; i++) {
            content[c_i] = NULL;
            setContent(c_i, tensors[idx]->content[i]);
            c_i += 1;
        }
    }
    name += ")";
}

Tensor::Tensor(const Tensor *origin, std::vector<unsigned int> idx) : dims(origin->dims)
{
    counter += 1;
    name = origin->name + "[";
    unsigned int start = 0;
    for (unsigned int i = 0; i < idx.size(); i++) {
        if (i > 0) {
            start *= dims[0];
            name += ",";
        }
        name += std::to_string(idx[i]);
        start += idx[i];
        dims.erase(dims.begin());
    }
    name += "]";
    for (unsigned int i = 0; i < dims.size(); i++) {
        start *= dims[i];
    }
    len = calculateLen();
    this->content = new Number*[len];
    for (unsigned int i = 0; i < len; i++) {
        content[i] = NULL;
        setContent(i, origin->content[start+i]);
    }
}

Tensor::Tensor(std::vector<unsigned int> dims, std::vector<FLOAT> values) : Tensor(dims)
{
    for (unsigned int i = 0; i < len; i++) {
        setContent(i, new Constant(values[i % values.size()]));
    }
    if (len % values.size() != 0)
        throw TensorException("values of initialization must be a divisor of the len of tensor", this);
}

Tensor::Tensor(std::vector<unsigned int> dims, std::vector<Number*> values) : Tensor(dims)
{
    for (unsigned int i = 0; i < len; i++) {
        setContent(i, values[i % values.size()]);
    }
    if (len % values.size() != 0)
        throw TensorException("values of initialization must be a divisor of the len of tensor", this);
}

Tensor::Tensor(FLOAT value) : Tensor(std::vector<unsigned int>(), std::vector<FLOAT>(1, value))
{
}

Tensor::Tensor(Number *value) : Tensor(std::vector<unsigned int>())
{
    setContent(0, value);
}

Tensor::Tensor(std::vector<unsigned int> dims, std::string group, Initializer *initializer) : Tensor(dims)
{
    if (initializer != NULL && initializer->getDims().size() > 0) {
        if (this->dims.size() != initializer->getDims().size()) {
            throw TensorException("wrong shape of initializer", this);
        }
        for (unsigned int i = 0; i < this->dims.size(); i++) {
            if (this->dims[i] != initializer->getDims()[i]) {
                throw TensorException("wrong shape of initializer", this);
            }
        }
    }
    for (unsigned int i = 0; i < len; i++) {
        std::string name = this->name + "_" + std::to_string(i);
        setContent(i, new Variable(group, name, initializer));
    }
}

Tensor::Tensor(std::string group, Initializer *initializer) : Tensor({}, group, initializer)
{
}

Tensor::~Tensor() {
    counter -= 1;
    for (unsigned int i = 0; i < len; i++) {
        unsetContent(i);
    }
    delete content;
}

void Tensor::setContent(unsigned int idx, Number *number) {
    if (content[idx]) {
        unsetContent(idx);
    }
    content[idx] = number;
    number->usedBy += 1;
}

void Tensor::unsetContent(unsigned int idx) {
    content[idx]->unset();
}

unsigned int Tensor::calculateLen() {
    if (dims.size() == 0)
        return 1;
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

Tensor *Tensor::shape() {
    if (dims.size() == 0)
        return new Tensor(1);
    Tensor *result = new Tensor(
        {(unsigned int)dims.size()},
        std::vector<FLOAT>(dims.begin(), dims.end())
    );
    result->name = name + "_shape";
    return result;
}

Tensor *Tensor::operator[](unsigned int idx) const {
    return get(idx);
}

Tensor *Tensor::get(unsigned int idx) const {
    if (dims.size() == 0)
        throw TensorException("can not access element of 0 dimensional tensor", this);
    if (idx >= dims[0])
        throw TensorException("Index out of bounds", this);
    return new Tensor(this, idx);
}

Tensor *Tensor::gather(std::vector<unsigned int> idx) const {
    if (dims.size() == 0)
        throw TensorException("can not access element of 0 dimensional tensor", this);
    if (idx.size() > dims.size())
        throw TensorException("Index out of bounds", this);
    for (unsigned int i = 1; i < idx.size(); i++) {
        if (idx[i] >= dims[i])
            throw TensorException("Index out of bounds", this);
    }
    return new Tensor(this, idx);
}

Tensor Tensor::getTmp(unsigned int idx) const {
    if (dims.size() == 0)
        throw TensorException("can not access element of 0 dimensional tensor", this);
    if (idx >= dims[0])
        throw TensorException("Index out of bounds", this);
    return Tensor(this, idx);
}

unsigned int Tensor::getAbsoluteIndex(std::vector<unsigned int> idx) const {
    if (idx.size() != dims.size()) {
        throw TensorException("Indexs doesn't match tensor dimension", this);
    }
    unsigned int index = idx[0];
    if (idx[0] >= dims[0])
        throw TensorException("Index out of bounds", this);
    for (unsigned int i = 1; i < dims.size(); i++) {
        if (idx[i] >= dims[i])
            throw TensorException("Index out of bounds", this);
        index *= dims[i];
        index += idx[i];
    }
    return index;
}

Number* Tensor::at(std::vector<unsigned int> idx) const {
    if (dims.size() == 0)
        throw TensorException("can not access element of 0 dimensional tensor", this);
    unsigned int index = getAbsoluteIndex(idx);
    return content[index];
}

Number& Tensor::asNumber() const {
    if (dims.size() > 0)
        throw TensorException("only scallar can be convert to number", this);
    return *content[0];
}

void Tensor::at(std::vector<unsigned int> idx, Number *number) {
    if (dims.size() == 0)
        throw TensorException("can not set element of 0 dimensional tensor", this);
    unsigned int index = getAbsoluteIndex(idx);
    setContent(index, number);
}

std::string Tensor::header() const {
    std::string shapeStr = " (shape=["; 
    for (unsigned int i = 0; i < dims.size(); i++) {
        if (i > 0)
            shapeStr += ",";
        shapeStr += std::to_string(dims[i]);
    }
    shapeStr += "])";
    return name + shapeStr;
}

std::string Tensor::toString(bool printGradient, int margin) const {

    std::string str = "";
    if (margin == 0) {
        str += header() + " : " + NL;
    }
    for (int i =0; i < margin; i++)
        str += " ";
    if (dims.size() == 0) {
        str += content[0]->toString(printGradient);
        return str;
    }
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
        str += getTmp(i).toString(printGradient, margin+1);
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

bool Tensor::sameShape(const Tensor &tensor) const {
    if (dims.size() != tensor.dims.size())
        return false;
    for (unsigned int i = 0; i < dims.size(); i++) {
        if (dims[i] != tensor.dims[i])
            return false;
    }
    return true;
}

Tensor *Tensor::add(const Tensor &tensor) const {
    static unsigned int c = 0;
    if (sameShape(tensor) == false && tensor.dims.size() > 0) {
        checkElementWiseOp(tensor.dims);
    }
    c+=1;
    Tensor *result = new Tensor(dims);
    for (unsigned int i = 0; i < len; i++) {
        result->setContent(i, new Addition(content[i], tensor.content[i % tensor.len]));
    }
    result->name = "add" + std::to_string(c);
    return result;
}

Tensor *Tensor::pow(const Tensor &tensor) const {
    static unsigned int c = 0;
    if (sameShape(tensor) == false && tensor.dims.size() > 0) {
        checkElementWiseOp(tensor.dims);
    }
    c+=1;
    Tensor *result = new Tensor(dims);
    result = new Tensor(dims);
    for (unsigned int i = 0; i < len; i++) {
        result->setContent(i, new Pow(content[i], tensor.content[i % tensor.len]));
    }
    result->name = "pow" + std::to_string(c);
    return result;
}

Tensor *Tensor::multiply(const Tensor &tensor) const {
    static unsigned int c = 0;
    if (sameShape(tensor) == false && tensor.dims.size() > 0) {
        checkElementWiseOp(tensor.dims);
    }
    c+=1;
    Tensor *result = new Tensor(dims);
    for (unsigned int i = 0; i < len; i++) {
        result->setContent(i, new Multiplication(content[i], tensor.content[i % tensor.len]));
    }
    result->name = "multiply" + std::to_string(c);
    return result;
}

Tensor *Tensor::inverse() const {
    static unsigned int c = 0;
    Tensor *result = new Tensor(dims);
    for (unsigned int i = 0; i < len; i++) {
        result->setContent(i, new Inverse(content[i]));
    }
    c+=1;
    result->name = "inverse" + std::to_string(c);
    return result;
}

Tensor *Tensor::negative() const {
    static unsigned int c = 0;
    Tensor *result = new Tensor(dims);
    for (unsigned int i = 0; i < len; i++) {
        result->setContent(i, new Negative(content[i]));
    }
    c+=1;
    result->name = "negative" + std::to_string(c);
    return result;
}

Tensor *Tensor::sigmoid() const {
    static unsigned int c = 0;
    Tensor *result = new Tensor(dims);
    for (unsigned int i = 0; i < len; i++) {
        result->setContent(i, new Sigmoid(content[i]));
    }
    c+=1;
    result->name = "sigmoid" + std::to_string(c);
    return result;
}

Tensor *Tensor::sin() const {
    static unsigned int c = 0;
    Tensor *result = new Tensor(dims);
    for (unsigned int i = 0; i < len; i++) {
        result->setContent(i, new Sinus(content[i]));
    }
    c+=1;
    result->name = "sinus" + std::to_string(c);
    return result;
}

Tensor *Tensor::cos() const {
    static unsigned int c = 0;
    Tensor *result = new Tensor(dims);
    for (unsigned int i = 0; i < len; i++) {
        result->setContent(i, new Cosinus(content[i]));
    }
    c+=1;
    result->name = "cosinus" + std::to_string(c);
    return result;
}

Tensor *Tensor::exp() const {
    static unsigned int c = 0;
    Tensor *result = new Tensor(dims);
    for (unsigned int i = 0; i < len; i++) {
        result->setContent(i, new Exp(content[i]));
    }
    c+=1;
    result->name = "exp" + std::to_string(c);
    return result;
}

Tensor *Tensor::log() const {
    static unsigned int c = 0;
    Tensor *result = new Tensor(dims);
    for (unsigned int i = 0; i < len; i++) {
        result->setContent(i, new Log(content[i]));
    }
    c+=1;
    result->name = "log" + std::to_string(c);
    return result;
}

Tensor *Tensor::vector_matmul(const Tensor &tensor) const {
    static unsigned int c = 0;

    if (dims.size() != 1 || tensor.dims.size() != 2)
        throw TensorException(
            "Matrix multiplication of a vector can only be done with 2 dimensional matrix and 1 dimensional vector",
            this, &tensor);
    if (dims[0] != tensor.dims[0])
        throw TensorException("Matrix multiplication dimensions doesn't match", this, &tensor);
    std::vector<unsigned int> new_dims;
    new_dims.push_back(tensor.dims[1]);
    Tensor *result = new Tensor(new_dims);
    for (unsigned int j = 0; j < tensor.dims[1]; j++) {
        std::vector<Number*> row;
        for (unsigned int k = 0; k < dims[0]; k++) {
            row.push_back(new Multiplication(at({k}), tensor.at({k, j})));
        }
        result->at({j}, new Sum(row));
    }
    c+=1;
    result->name = "vmatmul" + std::to_string(c);
    return result;
}

Tensor *Tensor::matmul(const Tensor &tensor) const {
    static unsigned int c = 0;

    if (tensor.dims.size() < 2)
        throw TensorException("Matrix multiplication can only be done with 2 dimensional matrix", this, &tensor);
    if (tensor.dims.size() > 2) {
        if (dims[0] != tensor.dims[0])
            throw TensorException("Unable to use matrix multiplication on tensors of different shapes", this, &tensor);
        std::vector<Tensor*> tensors;
        for (int idx = 0; idx < dims[0]; idx++) {
           tensors.push_back(get(idx)->matmul(*tensor.get(idx)));
        }
        return new Tensor(tensors);
    }
    if (dims.size() > 2) {
        std::vector<Tensor*> tensors;
        for (int idx = 0; idx < dims[0]; idx++) {
           tensors.push_back(get(idx)->matmul(tensor));
        }
        return new Tensor(tensors);
    }
    if (dims.size() == 1) {
        return vector_matmul(tensor);
    }
    if (dims.size() != 2)
        throw TensorException("Matrix multiplication can only be done with 2 dimensional matrix", this, &tensor);
    if (dims[1] != tensor.dims[0])
        throw TensorException("Matrix multiplication dimensions doesn't match", this, &tensor);
    Tensor *result = new Tensor({dims[0], tensor.dims[1]});
    for (unsigned int i = 0; i < dims[0]; i++) {
        for (unsigned int j = 0; j < tensor.dims[1]; j++) {
            std::vector<Number*> row;
            for (unsigned int k = 0; k < dims[1]; k++) {
                row.push_back(new Multiplication(at({i, k}), tensor.at({k, j})));
            }
            result->at({i,j}, new Sum(row));
        }
    }
    c+=1;
    result->name = "matmul" + std::to_string(c);
    return result;
}


Tensor *Tensor::minorXY(unsigned int X, unsigned int Y) const {
    static unsigned int c = 0;

    if (dims.size() != 2)
        throw TensorException("Matrix minor can only be done with 2 dimensional matrix", this);
    if (X >= dims[1] || Y >= dims[0])
        throw TensorException("Matrix minor indexs out limit", this);
    Tensor *result = new Tensor({dims[0]-1, dims[1]-1});
    unsigned x = 0;
    unsigned y = 0;
    for (unsigned int i = 0; i < dims[1]; i++) {
        if (i != X) {
            y = 0;
            for (unsigned int j = 0; j < dims[0]; j++) {
                if (j != Y) {
                    result->at({y,x}, at({j,i}));
                    y++;
                }
            }
            x++;
        }
    }
    c+=1;
    result->name = "minor_" + std::to_string(c);
    return result;
}

Tensor *Tensor::minorMatrix() const {
    if (dims.size() != 2)
        throw TensorException("minor matrix can only be build from 2 dimensional matrix", this);
    Tensor *result = new Tensor(dims);
    for (unsigned int i = 0; i < dims[1]; i++) {
        for (unsigned int j = 0; j < dims[0]; j++) {
            Number *n = &minorXY(i, j)->determinant()->asNumber();
            result->at({j,i}, n);
        }
    }
    return result;
}

Tensor *Tensor::matinv() const {
    if (dims.size() != 2)
        throw TensorException("Matrix inverse can only be build from 2 dimensional matrix", this);
    if (dims[0] != dims[1])
        throw TensorException("Matrix inverse can only be done with square matrix", this);
    Tensor *idet = determinant()->inverse();
    Tensor *mask = new Tensor(dims);
    Tensor *result = minorMatrix();
    for (unsigned int i = 0; i < dims[1]; i++) {
        for (unsigned int j = 0; j < dims[0]; j++) {
            if ((i+j) % 2 == 1) {
                mask->at({j,i}, new Constant(-1));
            } else {
                mask->at({j,i}, new Constant(1));
            }
        }
    }
    return result->multiply(*mask)->transpose({1,0})->multiply(*idet);
}


std::vector<unsigned int>    getDimIdx(std::vector<unsigned int> dims, unsigned int idx) {
    std::vector<unsigned int> didx;
    for (unsigned int I = 0; I< dims.size(); I++) {
        unsigned int i = dims.size() - (I+1);
        if (idx == 0) {
            didx.push_back(0);
        } else {
            unsigned int tmp = idx % dims[i];
            didx.push_back(tmp);
            idx /= dims[i];
        }
    }
    std::reverse(didx.begin(), didx.end());
    return didx;
}

Tensor *Tensor::transpose(std::vector<unsigned int>permutations) const {
    if (dims.size() != permutations.size())
        throw TensorException("Tensor transpose permutations must be equals to the tensor dimensions", this);
    std::vector<unsigned int> newdims;
    for (unsigned int i = 0; i < permutations.size(); i++) {
        if (permutations[i] >= dims.size())
            throw TensorException("Tensor transpose permutation out of original dimension", this);
        for (unsigned int j = 0; j < i; j++) {
            if (permutations[i] == permutations[j])
                throw TensorException("Tensor transpose permutation used twice", this);
        }
        newdims.push_back(dims[permutations[i]]);
    }
    Tensor *result = new Tensor(newdims);
    for (unsigned int i = 0; i < len; i++) {
        std::vector<unsigned int> didx = getDimIdx(dims, i);
        std::vector<unsigned int> nidx;
        for (unsigned int j = 0; j < didx.size(); j++) {
            nidx.push_back(didx[permutations[j]]);
        }
        result->at(nidx, content[i]);
    }
    return result;
}

Tensor *Tensor::determinant() const {
    static unsigned int c = 0;

    if (dims.size() != 2)
        throw TensorException("Matrix determinant can only be done with 2 dimensional matrix", this);
    if (dims[0] != dims[1])
        throw TensorException("Matrix determinant can only be done with square matrix", this);
    Number *determinant;
    if (dims[0] == 2) {
        determinant = new Substraction(new Multiplication(at({0,0}), at({1,1})), new Multiplication(at({0,1}), at({1,0})));
    }
    else {
        determinant = new Multiplication(at({0,0}), &minorXY(0,0)->determinant()->asNumber());
        for (unsigned int i = 1; i < dims[1]; i++) {
            Number *xminor = new Multiplication(at({0,i}), &minorXY(i,0)->determinant()->asNumber());
            if (i%2 == 0) {
                determinant = new Addition(determinant, xminor);
            } else {
                determinant = new Substraction(determinant, xminor);
            }
        }
    }
    Tensor *result =  new Tensor(determinant);
    c+=1;
    result->name = "determinant" + std::to_string(c);
    return result;
}

Tensor *Tensor::sum() const {
    static unsigned int c = 0;
    std::vector<Number*> vector;
    for (unsigned int i = 0; i < len; i++) {
        vector.push_back(content[i]);
    }
    Sum *sum = new Sum(vector);
    c+=1;
    Tensor *result = new Tensor(sum);
    result->name = "sum" + std::to_string(c);
    return result;
}

Tensor *Tensor::count() const {
    static unsigned int c = 0;
    std::vector<Number*> vector;
    for (unsigned int i = 0; i < len; i++) {
        vector.push_back(content[i]);
    }
    Count *count = new Count(vector);
    c+=1;
    Tensor *result = new Tensor(count);
    result->name = "count" + std::to_string(c);
    return result;
}

Tensor *Tensor::percentile(FLOAT percent) const {
    static unsigned int c = 0;
    std::vector<Number*> vector;
    for (unsigned int i = 0; i < len; i++) {
        vector.push_back(content[i]);
    }
    Percentile *percentile = new Percentile(vector, percent);
    c+=1;
    Tensor *result = new Tensor(percentile);
    result->name = std::to_string(percent) + "percentile" + std::to_string(c);
    return result;
}


Tensor *Tensor::std() const {
    Tensor *squareDiff = substract(*mean())->powRaw(2);
    return squareDiff->sum()->divide(*count())->powRaw(0.5);
}

void Tensor::gradientUpdate() {
    if (dims.size() > 0)
        throw TensorException("gradient update can only be done with a scalar", this);
    asNumber().calculateGradient();
}

void Tensor::gradientReinit() {
    if (dims.size() > 0)
        throw TensorException("gradient reinit can only be done with a scallar", this);
    asNumber().reinitGradient();
}

void Tensor::gradientChecking(std::string group) {
    if (dims.size() > 0)
        throw TensorException("gradient checking can only be done with a scallar", this);
    asNumber().checkAllGradient(group);
}

void Tensor::checkElementWiseOp(const std::vector<unsigned int> op_dims) const {
    if (op_dims.size() == 0) return;
    for (int i = 1; i <= op_dims.size(); i++) {
        int op_idx = op_dims.size() - i;
        int self_idx = dims.size() - i;
        if (self_idx < 0 || dims[self_idx] != op_dims[op_idx]) {
            throw TensorException("size for element wise operation doesn't match", this);
        }
    }
}

Tensor *Tensor::foreach(unsigned int from_dim, Tensor *(Tensor::*op)() const) const {
     if (from_dim == 0) {
        return (*this.*op)();
     }
     std::vector<Tensor*> tensors;
     for (int idx = 0; idx < dims[0]; idx++) {
        tensors.push_back(get(idx)->foreach(from_dim -1, op));
     }
     return new Tensor(tensors);
}

Tensor *Tensor::get(Tensor *indexs) const {
    if (indexs->dims.size() != dims.size() - 1)
       throw TensorException("get indexs must have the same shape than the original tensor minus the last dimension", this);
    if (dims.size() == 1) {
        if (indexs->len != 1) {
            throw TensorException("get indexs error", this);
        }
        std::vector<Number*> vector;
        for (int i = 0; i < dims[0]; i++) {
            vector.push_back(content[i]);
        }
        return new Tensor(new Get(vector, indexs->content[0]));
    }
    if (indexs->dims[0] != dims[0])
       throw TensorException("get indexs must have the same shape than the original tensor minus the last dimension", this);
     std::vector<Tensor*> tensors;
     for (int idx = 0; idx < dims[0]; idx++) {
        tensors.push_back(get(idx)->get(indexs->get(idx)));
     }
     return new Tensor(tensors);
}
