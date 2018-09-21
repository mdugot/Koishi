#include "utils.h"
#include "operation/variable.h"
#include "operation/addition.h"
#include "operation/multiplication.h"
#include "operation/sigmoid.h"
#include "operation/pow.h"
#include "operation/inverse.h"
#include "operation/sum.h"
#include "tensor/tensor.h"

void test();

int main(int argc, char **argv) {
	(void)argc;
	(void)argv;
    try {
        test();
        return 0;
    } catch (const std::exception &e) {
        DEBUG << "ERROR : " << e.what() << NL;
        return 1;
    }
}

void test() {
	DEBUG << "testing tensor lib" << NL;

    Tensor m1({2,3}, {1,2,3,3,2,1});
    Tensor m2({3,2}, {10,100,-1,-2,2,5});
    Tensor m3({3,9}, {
        111,112,113,121,122,123,131,132,133,
        211,212,213,221,222,223,231,232,233,
        311,312,313,321,322,323,331,332,333,
    });
    Tensor m4({3,1}, {-1,0,1});
    DEBUG << m1 << NL;
    DEBUG << m2 << NL;
    DEBUG << m3 << NL;
    DEBUG << m1.matmul(m2) << NL;
    DEBUG << m1.matmul(m3) << NL;
    DEBUG << m1.matmul(m4) << NL;
}
