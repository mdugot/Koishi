#ifndef TENSORS_H
#define TENSORS_H
#include "tensor/tensor.h"

class Tensors {

    private:
        std::vector<Tensor*> content;

        template<typename F>
        Tensors* forEach(F &functor) const {
            Tensors* result = new Tensors();
            for (unsigned int i = 0; i < content.size(); i++) {
                result->append(functor(content[i]));
            }
            return result;
        }

        template<typename F>
        Tensors* forEachPair(const Tensors *tensors, F &functor) const {
            if (content.size() != tensors->content.size())
                throw TensorException("the two list of tensors must have the same length");
            Tensors* result = new Tensors();
            for (unsigned int i = 0; i < content.size(); i++) {
                result->append(functor(content[i], tensors->content[i]));
            }
            return result;
        }

    public:
        Tensors(const Tensor *from, unsigned int splitAxis);
        Tensors();
        ~Tensors();

        std::string toString();
        void append(Tensor *tensor);
        Tensor* get(unsigned int i) const;
        Tensor* merge(std::vector<unsigned int>dims);
        inline unsigned int size() const {return content.size();}
        #ifdef PYTHON_WRAPPER
        inline std::string __str__() {return toString();}
        inline Tensor *mergeFromList(list& dims) {return merge(listToVector<unsigned int>(dims));}
        #endif

        inline Tensors* divideRaw(FLOAT value) const {
            auto f = [value](Tensor const*t){return t->divideRaw(value);};
            return forEach(f);}
        inline Tensors* divideTensor(const Tensor *tensor) const {
            auto f = [tensor](Tensor const*t){return t->divide(*tensor);};
            return forEach(f);}
        inline Tensors* divide(const Tensors *tensors) const {
            auto f = [](Tensor const*t1, Tensor const*t2){return t1->divide(*t2);};
            return forEachPair(tensors, f);}
        inline Tensors* multiplyRaw(FLOAT value) const {
            auto f = [value](Tensor const*t){return t->multiplyRaw(value);};
            return forEach(f);}
        inline Tensors* multiplyTensor(const Tensor *tensor) const {
            auto f = [tensor](Tensor const*t){return t->multiply(*tensor);};
            return forEach(f);}
        inline Tensors* multiply(const Tensors *tensors) const {
            auto f = [](Tensor const*t1, Tensor const*t2){return t1->multiply(*t2);};
            return forEachPair(tensors, f);}
        inline Tensors* substractRaw(FLOAT value) const {
            auto f = [value](Tensor const*t){return t->substractRaw(value);};
            return forEach(f);}
        inline Tensors* substractTensor(const Tensor *tensor) const {
            auto f = [tensor](Tensor const*t){return t->substract(*tensor);};
            return forEach(f);}
        inline Tensors* substract(const Tensors *tensors) const {
            auto f = [](Tensor const*t1, Tensor const*t2){return t1->substract(*t2);};
            return forEachPair(tensors, f);}
        inline Tensors* addRaw(FLOAT value) const {
            auto f = [value](Tensor const*t){return t->addRaw(value);};
            return forEach(f);}
        inline Tensors* addTensor(const Tensor *tensor) const {
            auto f = [tensor](Tensor const*t){return t->add(*tensor);};
            return forEach(f);}
        inline Tensors* add(const Tensors *tensors) const {
            auto f = [](Tensor const*t1, Tensor const*t2){return t1->add(*t2);};
            return forEachPair(tensors, f);}
        inline Tensors* sigmoid() const {
            auto f = [](Tensor const*t){return t->sigmoid();};
            return forEach(f);}
        inline Tensors* mean() const {
            auto f = [](Tensor const*t){return t->mean();};
            return forEach(f);}
        inline Tensors* max() const {
            auto f = [](Tensor const*t){return t->max();};
            return forEach(f);}
        inline Tensors* min() const {
            auto f = [](Tensor const*t){return t->min();};
            return forEach(f);}
        inline Tensors* range() const {
            auto f = [](Tensor const*t){return t->range();};
            return forEach(f);}

};

#endif
