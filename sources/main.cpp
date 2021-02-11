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

void maintest();

int main(int argc, char **argv) {
    try {
        maintest();
        return 0;
    } catch (const std::exception &e) {
        DEBUG << "ERROR : " << e.what() << NL;
        return 1;
    }
}

void maintest() {
    Tensor t(42.0);
    DEBUG << t << NL << NL;

}
