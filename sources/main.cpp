#include "utils.h"

int main(int argc, char **argv) {
	(void)argc;
	(void)argv;
	DEBUG << "testing tensor lib" << NL;

	Shape shape({3,4});
	Tensor t1(shape, 3);
	Tensor t2(shape, 2);
	
	Tensor r1 = Tensor.add(t1, t2);
	Tensor r2 = Tensor.add(Tensor.sigmoid(r1), -1);

	DEBUG << r2 << NL;
    
}
