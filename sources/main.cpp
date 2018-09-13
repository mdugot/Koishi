#include "utils.h"
#include "operation/variable.h"
#include "operation/addition.h"
#include "operation/multiplication.h"
#include "operation/sigmoid.h"

int main(int argc, char **argv) {
	(void)argc;
	(void)argv;
	DEBUG << "testing tensor lib" << NL;

//	Shape shape({3,4});
//	Tensor t1(shape, 3);
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
    DEBUG << "result : " << sig.eval() << NL;
    DEBUG << "derivate 1 : " << sig.derivate(&v1) << NL;
    DEBUG << "derivate 2 : " << sig.derivate(&v2) << NL;
    
}
