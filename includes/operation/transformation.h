#include "utils.h"

virtual class Transformation : Number {

    protected:
        Operation *base;

    public:
        Transformation(Operation *previous);
}

