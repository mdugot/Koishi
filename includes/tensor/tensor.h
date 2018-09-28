#ifndef TENSOR_H
#define TENSOR_H
#include "utils.h"
#include "wrapper/wrapperTools.h"
#include "tensor/tensorException.h"
#include "operation/number.h"
#include "initializer/initializer.h"

class Sum;
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
        static unsigned int count;

        Tensor(FLOAT value);
        Tensor(Number* value);
        Tensor(std::vector<unsigned int> dims, std::vector<FLOAT> values);
        Tensor(std::string group, Initializer &initializer);
        Tensor(std::vector<unsigned int> dims, std::string group, Initializer &initializer);
        Tensor(const Tensor *origin, unsigned int idx);
        ~Tensor();

        inline void setName(std::string str) {name = str;}
        inline std::string getName() {return name;}
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
        Tensor *add(const Tensor &tensor) const;
        Tensor *pow(const Tensor &tensor) const;
        Tensor *multiply(const Tensor &tensor) const;
        Tensor *inverse() const;
        Tensor *sigmoid() const;
        Tensor *sum() const;
        Tensor *matmul(const Tensor &tensor) const;

        void gradientReinit();
        void gradientUpdate();
        void gradientChecking(std::string group);


        #ifdef PYTHON_WRAPPER
        inline std::string __str__() {return toString();}
        inline void printGradient() {OUT << toString(true) << NL;}
        Tensor(boost::python::list &dims, boost::python::list &values);
        Tensor(std::string group, InitializerWrapper &wrap);
        Tensor(boost::python::list &dims, std::string group, InitializerWrapper &wrap);
        #endif
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

#endif
