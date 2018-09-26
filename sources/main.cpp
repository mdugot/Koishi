#include "utils.h"
#include "operation/variable.h"
#include "operation/addition.h"
#include "operation/multiplication.h"
#include "operation/sigmoid.h"
#include "operation/pow.h"
#include "operation/inverse.h"
#include "operation/sum.h"
#include "tensor/tensor.h"
#include "initializer/uniform.h"

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

//    Variable *v = new Variable("test", "test", 0.5);
//    Inverse *inv = new Inverse(v);
//    DEBUG << "VALUE : " << inv->toString() << NL;
//    DEBUG << "CHECKING : " << inv->gradientChecking(v) << NL;

    Uniform init(-1,1);
    Tensor m1({2,3}, "v1", init);
    Tensor m2({3,2}, "v2", init);
    Tensor mm = m1.matmul(m2);
    Tensor inv = mm.inverse();
    Tensor sig = inv.sigmoid();
    Sum sum = sig.sum();

    Variable::load("./save.txt");
//    init.init();

    DEBUG << "SUM : " << sum.toString() << NL;
    sig.calculateGradient();
    DEBUG << m1.toString(true) << NL;
    DEBUG << m2.toString(true) << NL;

    sum.checkAllGradient("v1");

//    Variable::save("./save.txt");


}
