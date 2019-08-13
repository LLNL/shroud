// wrapns_outer.cpp
// This is generated code, do not edit
#include "wrapns_outer.h"
#include "namespace.hpp"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

// void One()
void NS_outer_one()
{
// splicer begin function.one
    outer::One();
    return;
// splicer end function.one
}

}  // extern "C"
