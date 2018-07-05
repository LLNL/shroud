// wraptemplates.cpp
// This is generated code, do not edit
#include "wraptemplates.h"
#include "typestemplates.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

void TEM_function_tu_0(int arg1, long arg2)
{
// splicer begin function.function_tu_0
    FunctionTU<int, long>(arg1, arg2);
    return;
// splicer end function.function_tu_0
}

void TEM_function_tu_1(float arg1, double arg2)
{
// splicer begin function.function_tu_1
    FunctionTU<float, double>(arg1, arg2);
    return;
// splicer end function.function_tu_1
}

// Release C++ allocated memory.
void TEM_SHROUD_memory_destructor(TEM_SHROUD_capsule_data *cap)
{
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
