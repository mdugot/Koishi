#include "tensor/tensor.h"
#include "operation/variable.h"
#include "operation/constant.h"
#include "operation/addition.h"
#include "operation/inverse.h"
#include "operation/sigmoid.h"
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

    Tensor::Tensor(std::string group, np::ndarray &a)
    : Tensor(
        getNumpyShape(a),
        group,
        NULL
    ){
        std::vector<FLOAT> values = numpyToVector(a);
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
        group,
        wrap.initializer
    ){}
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

Tensor::Tensor(std::vector<unsigned int> dims, std::vector<FLOAT> values) : Tensor(dims)
{
    for (unsigned int i = 0; i < len; i++) {
        setContent(i, new Constant(values[i % values.size()]));
    }
    if (len % values.size() != 0)
        throw TensorException("values of initialization must be a divisor of the len of tensor", this);
}

Tensor::Tensor(FLOAT value) : Tensor(std::vector<unsigned int>(), {value})
{
}

Tensor::Tensor(Number *value) : Tensor(std::vector<unsigned int>())
{
    setContent(0, value);
}

Tensor::Tensor(std::vector<unsigned int> dims, std::string group, Initializer *initializer) : Tensor(dims)
{
    for (unsigned int i = 0; i < len; i++) {
        std::string name = this->name + "_" + std::to_string(i);
        setContent(i, new Variable(group, name, initializer));
    }
}

Tensor::Tensor(std::string group, Initializer *initializer) : Tensor({}, group, initializer)
{
}

Tensor::Tensor(const Tensor *origin, unsigned int idx) : dims(origin->dims)
{
    counter += 1;
    name = origin->name + "[" + std::to_string(idx) + "]";
    dims.erase(dims.begin());
    len = calculateLen();
    this->content = new Number*[len];
    for (unsigned int i = 0; i < len; i++) {
        setContent(i, origin->content[len*idx + i]);
    }
}

Tensor::~Tensor() {
    counter -= 1;
    for (unsigned int i = 0; i < len; i++)
        unsetContent(i);
    delete content;
}

void Tensor::setContent(unsigned int idx, Number *number) {
    if (content[idx])
        unsetContent(idx);
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
    if (sameShape(tensor) == false && tensor.dims.size() > 0)
        throw TensorException("can not add tensor of different shapes", this, &tensor);
    c+=1;
    Tensor *result = new Tensor(dims);
    if (tensor.dims.size() == 0) {
        for (unsigned int i = 0; i < len; i++) {
            result->setContent(i, new Addition(content[i], tensor.content[0]));
        }
    } else {
        for (unsigned int i = 0; i < len; i++) {
            result->setContent(i, new Addition(content[i], tensor.content[i]));
        }
    }
    result->name = "add" + std::to_string(c);
    return result;
}

Tensor *Tensor::pow(const Tensor &tensor) const {
    static unsigned int c = 0;
    if (sameShape(tensor) == false && tensor.dims.size() > 0)
        throw TensorException("can not pow tensor of different shapes", this, &tensor);
    c+=1;
    Tensor *result = new Tensor(dims);
    if (tensor.dims.size() == 0) {
        for (unsigned int i = 0; i < len; i++) {
            result->setContent(i, new Pow(content[i], tensor.content[0]));
        }
    } else {
        for (unsigned int i = 0; i < len; i++) {
            result->setContent(i, new Pow(content[i], tensor.content[i]));
        }
    }
    result->name = "pow" + std::to_string(c);
    return result;
}

Tensor *Tensor::multiply(const Tensor &tensor) const {
    static unsigned int c = 0;
    if (sameShape(tensor) == false && tensor.dims.size() > 0)
        throw TensorException("can not multiply tensor of different shapes", this, &tensor);
    c+=1;
    Tensor *result = new Tensor(dims);
    if (tensor.dims.size() == 0) {
        for (unsigned int i = 0; i < len; i++) {
            result->setContent(i, new Multiplication(content[i], tensor.content[0]));
        }
    } else {
        for (unsigned int i = 0; i < len; i++) {
            result->setContent(i, new Multiplication(content[i], tensor.content[i]));
        }
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

Tensor *Tensor::gaussianLikehood(const Tensor &mean, const Tensor &std) const {
    static unsigned int c = 0;

    if (mean.dims.size() > 0)
        throw TensorException("gaussian likehood mean must be a scallar", this, &mean);
    if (std.dims.size() > 0)
        throw TensorException("gaussian likehood standard deviation must be a scallar", this, &std);
    Tensor *result = new Tensor(dims);
    for (unsigned int i = 0; i < len; i++) {
        Number *factor = new Inverse(new Multiplication(
            &std.asNumber(),
            new Pow(
                new Constant(2*M_PI),
                new Constant(0.5)
            )
        ));
        Number *power = new Multiplication(
            new Constant(-0.5),
            new Pow(
                new Division(
                    new Substraction(content[i], &mean.asNumber()),
                    &std.asNumber()
                ),
                new Constant(2)
            )
        );
        (void)factor;
        (void)power;
        Number *gauss = new Multiplication(factor, new Pow(new Constant(M_E), power));
        //Number *gauss = power;
        result->setContent(i, gauss);
    }
    c+=1;
    result->name = "gaussian" + std::to_string(c);
    return result;
}

Tensor *Tensor::matmul(const Tensor &tensor) const {
    static unsigned int c = 0;

    if (dims.size() != 2 || tensor.dims.size() != 2)
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
