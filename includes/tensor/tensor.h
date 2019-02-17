#ifndef TENSOR_H
#define TENSOR_H
#include "utils.h"
#include "wrapper/wrapperTools.h"
#include "tensor/tensorException.h"
#include "operation/number.h"
#include "operation/constant.h"
#include "initializer/initializer.h"

class Sum;
class Tensors;
#ifdef PYTHON_WRAPPER
class InitializerWrapper;
#endif

class Tensor {

    private:
        Tensor(std::vector<unsigned int> dims);

        std::vector<unsigned int> dims;
        unsigned int len;
        Number** content;

        std::string name;
        Tensor getTmp(unsigned int idx) const;


    public:
        static unsigned int counter;

        Tensor(FLOAT value);
        Tensor(Number* value);
        Tensor(std::vector<unsigned int> dims, std::vector<FLOAT> values);
        Tensor(std::string group, Initializer *initializer);
        Tensor(std::vector<unsigned int> dims, std::string group, Initializer *initializer);
        Tensor(const Tensor *origin, unsigned int idx);
        Tensor(const Tensor *origin, std::vector<unsigned int> idx);
        Tensor(const Tensors *origin, std::vector<unsigned int> dims);
        ~Tensor();

        inline void setName(std::string str) {name = str;}
        inline std::string getName() {return name;}
        inline std::vector<unsigned int> getDims() const {return dims;}
        void setContent(unsigned int idx, Number *number);
        void unsetContent(unsigned int idx);
        unsigned int calculateLen();
        void calculateGradient();
        std::string toString(bool printGradient = false, int margin = 0) const;
        std::string header() const;
        bool equals(Tensor &tensor);
        bool sameShape(const Tensor &tensor) const;
        unsigned int getAbsoluteIndex(std::vector<unsigned int>idx) const;
        Number* at(std::vector<unsigned int>idx) const;
        void at(std::vector<unsigned int>idx, Number* number);
        Number& asNumber() const;

        Tensor *shape();
        Tensor *operator[](unsigned int idx) const;
        Tensor *get(unsigned int idx) const;
        Tensor *gather(std::vector<unsigned int> idx) const;
        Tensors *split(unsigned int splitAxis) const;
        Tensor *add(const Tensor &tensor) const;
        Tensor *pow(const Tensor &tensor) const;
        Tensor *multiply(const Tensor &tensor) const;
        Tensor *inverse() const;
        Tensor *negative() const;
        Tensor *sigmoid() const;
        Tensor *sum() const;
        Tensor *count() const;
        Tensor *std() const;
        Tensor *matmul(const Tensor &tensor) const;
        Tensor *minor(unsigned int X, unsigned int Y) const;
        Tensor *determinant() const;
        Tensor *minorMatrix() const;
        Tensor *matinv() const;
        Tensor *transpose(std::vector<unsigned int>permutations) const;
        Tensor *percentile(FLOAT percent) const;

        inline Tensor *addRaw(FLOAT rawValue) const {return add(Tensor(rawValue));}
        inline Tensor *powRaw(FLOAT rawValue) const {return pow(Tensor(rawValue));}
        inline Tensor *multiplyRaw(FLOAT rawValue) const {return multiply(Tensor(rawValue));}
        inline Tensor *divide(const Tensor &tensor) const {return multiply(*tensor.inverse());}
        inline Tensor *divideRaw(FLOAT rawValue) const {return divide(Tensor(rawValue));}
        inline Tensor *substract(const Tensor &tensor) const {return add(*tensor.multiply(-1));}
        inline Tensor *substractRaw(FLOAT rawValue) const {return substract(Tensor(rawValue));}
        inline Tensor *max() const {return percentile(100);}
        inline Tensor *min() const {return percentile(0);}
        inline Tensor *mean() const {return sum()->divide(*count());}

        void gradientReinit();
        void gradientUpdate();
        void gradientChecking(std::string group);


        #ifdef PYTHON_WRAPPER
        inline std::string __str__() {return toString();}
        inline void printGradient() {OUT << toString(true) << NL;}
        Tensor(boost::python::numpy::ndarray &values);
        Tensor(std::string group, InitializerWrapper &wrap);
        Tensor(boost::python::list &dims, std::string group, InitializerWrapper &wrap);
        Tensor(std::string group, FLOAT value);
        Tensor(std::string group, boost::python::numpy::ndarray &a);
        boost::python::object evalForPython();
        Tensor *transposeFromList(list& permutations);
        Tensor *gatherFromList(list& permutations);
        #endif
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

#endif
