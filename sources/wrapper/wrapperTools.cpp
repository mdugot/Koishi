#ifdef PYTHON_WRAPPER
#include "wrapper/wrapperTools.h"
#include "initializer/uniform.h"
#include "operation/numberException.h"

InitializerWrapper::InitializerWrapper(std::string type, FLOAT min, FLOAT max) {
    if (type == "uniform")
        initializer = new Uniform(min, max);
    else
        throw NumberException("Unknown initializer : " + type);
}

void InitializerWrapper::init() {
    initializer->init();
}

#endif
