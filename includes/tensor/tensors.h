#ifndef TENSORS_H
#define TENSORS_H
#include "tensor/tensor.h"

class Tensors {

    private:
        std::vector<Tensor*> content;

//        template<typename Functor>
//        Tensors* forEach(Functor functor)

//        template<typename Functor>
//        Tensors* forEachPair(Functor functor)

    public:
        Tensors(const Tensor *from, unsigned int splitAxis);

        std::string toString();
        #ifdef PYTHON_WRAPPER
        inline std::string __str__() {return toString();}
        #endif

//        Tensors* addTensor(const Tensor &tensor) const {return forEach([](Tensor const&t) {t.add(tensor)});}
//        Tensors* addRawTensor(FLOAT value) const {return forEach([this](Tensor const&t) {t.addRaw(value)});}
//        Tensors* add(const Tensors &tensors) const {return forEachPair([this, tensors](Tensor const&t1, Tensor const&t2) {t1.add(t2)});}

};

#endif
