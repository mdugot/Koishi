#include "utils.h"

virtual class Number {

    protected:

    public:
        Number();
        virtual float eval();
        virtual float derivate(Operation *respectOf);
}
