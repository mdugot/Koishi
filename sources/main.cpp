#include "utils.h"
#include "operation/variable.h"
#include "operation/addition.h"
#include "operation/multiplication.h"
#include "operation/sigmoid.h"
#include "operation/pow.h"
#include "operation/inverse.h"
#include "tensor/tensor.h"

int main(int argc, char **argv) {
	(void)argc;
	(void)argv;
	DEBUG << "testing tensor lib" << NL;

	Tensor t1({5,3,4}, 3);
    DEBUG << t1;
//	Tensor t2(shape, 2);
//	
//	Tensor r1 = Tensor.add(t1, t2);
//	Tensor r2 = Tensor.add(Tensor.sigmoid(r1), -1);
//
//	DEBUG << r2 << NL;
    
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
