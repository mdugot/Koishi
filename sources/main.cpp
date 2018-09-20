#include "utils.h"
#include "operation/variable.h"
#include "operation/addition.h"
#include "operation/multiplication.h"
#include "operation/sigmoid.h"
#include "operation/pow.h"
#include "operation/inverse.h"
#include "tensor/tensor.h"
#include "tensor/tensorIndexException.h"

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

	Tensor<Tensor<Number>> t1({3,2,4}, 3);
	Tensor<Number> t2({3,2,4});
    Tensor<Number> shape = t1.shape();
    DEBUG << t2.toString() << NL;
    DEBUG << shape.toString() << NL;
//    DEBUG << shape.equals(t2) << NL;

    Variable v1(3);
    Variable v2(-2);
    Multiplication mul(&v1, &v2);
    Sigmoid sig(&mul);
    Pow pow(&v1, &v2);
    Inverse inv(&mul);
    Multiplication mul2(&sig, &inv);
    Addition add(&pow, &mul2);

    Number &result = add;
    DEBUG << "result : " << result.eval() << NL;
    DEBUG << "derivate 1 : " << result.derivate(&v1) << NL;
    DEBUG << "derivate 2 : " << result.derivate(&v2) << NL;
    result.reinitGradient();
    result.calculateGradient();
    DEBUG << "gradient 1 : " << v1.getGradient() << NL;
    DEBUG << "gradient 2 : " << v2.getGradient() << NL;
    DEBUG << "checking 1 : " << result.gradientChecking(&v1) << NL;
    DEBUG << "checking 2 : " << result.gradientChecking(&v2) << NL;
}
