#include "initializer/initializer.h"
#include "operation/variable.h"

std::list<Initializer*> Initializer::all;

void Initializer::initializeAll() {
    for (Initializer* init : all) {
        init->init();
    }
}

Initializer::Initializer()
{
    all.push_back(this);
}

Initializer::Initializer(std::vector<unsigned int> dims) : Initializer()
{
    this->dims = dims;
}

Initializer::~Initializer()
{
    all.remove(this);
    for (Variable* v : variables) {
        v->setInitializer(NULL);
    }
}

void Initializer::add(Variable *v) {
    variables.push_back(v);
}

void Initializer::remove(Variable *v) {
    variables.remove(v);
}
