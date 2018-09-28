#ifdef PYTHON_WRAPPER
#include "wrapper/wrapperTools.h"
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

InitializerWrapper *getUniformInitializer(FLOAT min, FLOAT max) {
    return new InitializerWrapper(new Uniform(min,max));
}

FeedWrapper *getFeedInitializer(list &values) {
    return new FeedWrapper(new Feed(listToVector<FLOAT>(values)));
}


#endif
