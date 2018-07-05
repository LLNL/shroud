// wrapvectorforint.cpp
// This is generated code, do not edit
#include "wrapvector.h"
#include <vector>

// splicer begin class.vector.CXX_definitions
// splicer end class.vector.CXX_definitions

extern "C" {

// splicer begin class.vector.C_definitions
// splicer end class.vector.C_definitions

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
