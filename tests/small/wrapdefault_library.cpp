// wrapdefault_library.cpp
// This is generated code, do not edit
// wrapdefault_library.cpp
#include "wrapdefault_library.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

void DEF_vector1_bufferify(const int * arg, long Narg)
{
// splicer begin function.vector1_bufferify
    const std::vector<int> SH_arg(arg, arg + Narg);
    vector1(SH_arg);
    return;
// splicer end function.vector1_bufferify
}

}  // extern "C"
