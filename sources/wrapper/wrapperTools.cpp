#ifdef PYTHON_WRAPPER
#include "wrapper/wrapperTools.h"
#include "tensor/tensor.h"
#include "initializer/uniform.h"
#include "initializer/feed.h"
#include "operation/numberException.h"

InitializerWrapper::InitializerWrapper(Initializer *initializer) :initializer(initializer)
{
}

InitializerWrapper::~InitializerWrapper() {
    delete initializer;
}

void InitializerWrapper::init() {
    initializer->init();
}

FeedWrapper::FeedWrapper(Feed *feeder) :InitializerWrapper(feeder), feeder(feeder)
{
}

FeedWrapper::~FeedWrapper() {
}

void FeedWrapper::feed(list &l) {
    feeder->feed(listToVector<FLOAT>(l));
}

void FeedWrapper::feedSimple(FLOAT value) {
    feeder->feed({value});
}

InitializerWrapper *getUniformInitializer(FLOAT min, FLOAT max) {
    return new InitializerWrapper(new Uniform(min,max));
}

InitializerWrapper *getFillInitializer(FLOAT value) {
    return new InitializerWrapper(new Fill(value));
}

FeedWrapper *getFeedInitializer(list &values) {
    return new FeedWrapper(new Feed(listToVector<FLOAT>(values)));
}

FeedWrapper *getSimpleFeedInitializer(FLOAT value) {
    return new FeedWrapper(new Feed({value}));
}

object Tensor::evalForPython() {
    if (dims.size() == 0)
        return object(content[0]->eval());
    list result;
    for (unsigned int i = 0; i < dims[0]; i++) {
        result.append(getTmp(i).evalForPython());
    }
    return result;
}


#endif
