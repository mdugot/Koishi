#ifndef TENSORS_H
#define TENSORS_H
#include "tensor/tensor.h"

class Tensors {

    private:
        std::vector<Tensor*> content;

        template<typename F>
        Tensors* forEach(F &functor) const {
            Tensors* result = new Tensors();
            (void)functor;
            for (unsigned int i = 0; i < content.size(); i++) {
                result->append(functor(content[i]));
            }
            return result;
        }

//        template<typename Functor>
//        Tensors* forEachPair(Functor functor)

    public:
        Tensors(const Tensor *from, unsigned int splitAxis);
        Tensors();
        ~Tensors();

        std::string toString();
        void append(Tensor *tensor);
        #ifdef PYTHON_WRAPPER
        inline std::string __str__() {return toString();}
        #endif

        inline Tensors* addRaw(FLOAT value) const {
            auto f = [value](Tensor const*t){return t->addRaw(value);};
            return forEach(f);}
        inline Tensors* addTensor(const Tensor *tensor) const {
            auto f = [tensor](Tensor const*t){return t->add(*tensor);};
            return forEach(f);}
        inline Tensors* sigmoid() const {
            auto f = [](Tensor const*t){return t->sigmoid();};
            return forEach(f);}

};

#endif
