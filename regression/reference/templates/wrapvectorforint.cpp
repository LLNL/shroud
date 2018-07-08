// wrapvectorforint.cpp
// This is generated code, do not edit
#include "wrapvector.h"
#include <stdlib.h>
#include <vector>

// splicer begin class.vector.CXX_definitions
// splicer end class.vector.CXX_definitions

extern "C" {

// splicer begin class.vector.C_definitions
// splicer end class.vector.C_definitions

TEM_vector TEM_vector_ctor()
{
// splicer begin class.vector.method.ctor
    std::vector *SHCXX_rv = new std::vector_0();
    TEM_vector SHC_rv;
    SHC_rv.addr = static_cast<void *>(SHCXX_rv);
    SHC_rv.idtor = 0;
    return SHC_rv;
// splicer end class.vector.method.ctor
}

void TEM_vector_dtor(TEM_vector_0 * self)
{
// splicer begin class.vector.method.dtor
    std::vector<int> *SH_this =
        static_cast<std::vector<int> *>(self->addr);
    delete SH_this;
    self->addr = NULL;
    return;
// splicer end class.vector.method.dtor
}

void TEM_vector_push_back_XXXX(TEM_vector_0 * self, const int value)
{
// splicer begin class.vector.method.push_back_XXXX
    std::vector<int> *SH_this =
        static_cast<std::vector<int> *>(self->addr);
    SH_this->push_back(value);
    return;
// splicer end class.vector.method.push_back_XXXX
}

}  // extern "C"
