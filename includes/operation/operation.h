#include "utils.h"

virtual class Operation : Number {

    protected:
        Operation *left;
        Operation *right;

    public:
        Operation(Operation *previous);
}
