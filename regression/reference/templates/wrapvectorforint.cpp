// wrapvectorforint.cpp
// This is generated code, do not edit
#include "wrapvector_int.h"
#include <stdlib.h>
#include <vector>

// splicer begin class.vector.CXX_definitions
// splicer end class.vector.CXX_definitions

extern "C" {

// splicer begin class.vector.C_definitions
// splicer end class.vector.C_definitions

TEM_vector_int TEM_vector_int_ctor()
{
// splicer begin class.vector.method.ctor
    std::vector<int> *SHCXX_rv = new std::vector<int>();
    TEM_vector_int SHC_rv;
    SHC_rv.addr = static_cast<void *>(SHCXX_rv);
    SHC_rv.idtor = 0;
    return SHC_rv;
// splicer end class.vector.method.ctor
}

void TEM_vector_int_dtor(TEM_vector_int * self)
{
// splicer begin class.vector.method.dtor
    std::vector<int> *SH_this =
        static_cast<std::vector<int> *>(self->addr);
    delete SH_this;
    self->addr = NULL;
    return;
// splicer end class.vector.method.dtor
}

void TEM_vector_int_push_back(TEM_vector_int * self, const int * value)
{
// splicer begin class.vector.method.push_back
    std::vector<int> *SH_this =
        static_cast<std::vector<int> *>(self->addr);
    SH_this->push_back(*value);
    return;
// splicer end class.vector.method.push_back
}

int * TEM_vector_int_at(TEM_vector_int * self, size_t n)
{
// splicer begin class.vector.method.at
    std::vector<int> *SH_this =
        static_cast<std::vector<int> *>(self->addr);
    int & SHC_rv = SH_this->at(n);
    return &SHC_rv;
// splicer end class.vector.method.at
}

}  // extern "C"
