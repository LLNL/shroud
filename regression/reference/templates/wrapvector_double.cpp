// wrapvector_double.cpp
// This is generated code, do not edit
#include "wrapvector_double.h"
#include <stdlib.h>
#include <vector>

// splicer begin class.vector.CXX_definitions
// splicer end class.vector.CXX_definitions

extern "C" {

// splicer begin class.vector.C_definitions
// splicer end class.vector.C_definitions

TEM_vector_double * TEM_vector_double_ctor(TEM_vector_double * SHC_rv)
{
// splicer begin class.vector.method.ctor
    std::vector<double> *SHCXX_rv = new std::vector<double>();
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end class.vector.method.ctor
}

void TEM_vector_double_dtor(TEM_vector_double * self)
{
// splicer begin class.vector.method.dtor
    std::vector<double> *SH_this =
        static_cast<std::vector<double> *>(self->addr);
    delete SH_this;
    self->addr = NULL;
    return;
// splicer end class.vector.method.dtor
}

void TEM_vector_double_push_back(TEM_vector_double * self,
    const double * value)
{
// splicer begin class.vector.method.push_back
    std::vector<double> *SH_this =
        static_cast<std::vector<double> *>(self->addr);
    SH_this->push_back(*value);
    return;
// splicer end class.vector.method.push_back
}

double * TEM_vector_double_at(TEM_vector_double * self, size_t n)
{
// splicer begin class.vector.method.at
    std::vector<double> *SH_this =
        static_cast<std::vector<double> *>(self->addr);
    double & SHC_rv = SH_this->at(n);
    return &SHC_rv;
// splicer end class.vector.method.at
}

}  // extern "C"
