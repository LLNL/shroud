// wraptemplates.cpp
// This is generated code, do not edit
#include "wraptemplates.h"
#include <stdlib.h>
#include <vector>
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
    void *ptr = cap->addr;
    switch (cap->idtor) {
    case 0:   // --none--
    {
        // Nothing to delete
        break;
    }
    case 1:   // std::vector<int>
    {
        std::vector<int> *cxx_ptr = 
            reinterpret_cast<std::vector<int> *>(ptr);
        delete cxx_ptr;
        break;
    }
    default:
    {
        // Unexpected case in destructor
        break;
    }
    }
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
