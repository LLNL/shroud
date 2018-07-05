// wrapuser.cpp
// This is generated code, do not edit
#include "wrapuser.h"

// splicer begin class.user.CXX_definitions
// splicer end class.user.CXX_definitions

extern "C" {

// splicer begin class.user.C_definitions
// splicer end class.user.C_definitions

void TEM_user_nested_double(TEM_user_0 * self, int arg1, double arg2)
{
// splicer begin class.user.method.nested_double
    user<int> *SH_this = static_cast<user<int> *>(self->addr);
    SH_this->nested<double>(arg1, arg2);
    return;
// splicer end class.user.method.nested_double
}

}  // extern "C"
