// wraptemplates.cpp
// This is generated code, do not edit
#include "wraptemplates.h"
#include <cstdlib>
#include <vector>
#include "templates.hpp"
#include "typestemplates.h"

#include "implworker1.hpp"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {

// splicer begin C_definitions
// splicer end C_definitions

/**
 * \brief Function template with two template parameters.
 *
 */
void TEM_function_tu_0(int arg1, long arg2)
{
// splicer begin function.function_tu_0
    FunctionTU<int, long>(arg1, arg2);
    return;
// splicer end function.function_tu_0
}

/**
 * \brief Function template with two template parameters.
 *
 */
void TEM_function_tu_1(float arg1, double arg2)
{
// splicer begin function.function_tu_1
    FunctionTU<float, double>(arg1, arg2);
    return;
// splicer end function.function_tu_1
}

/**
 * \brief Function which uses a templated T in the implemetation.
 *
 */
int TEM_use_impl_worker_internal_ImplWorker1()
{
// splicer begin function.use_impl_worker_internal_ImplWorker1
    int SHC_rv = UseImplWorker<internal::ImplWorker1>();
    return SHC_rv;
// splicer end function.use_impl_worker_internal_ImplWorker1
}

// Release library allocated memory.
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
    case 2:   // std::vector<double>
    {
        std::vector<double> *cxx_ptr = 
            reinterpret_cast<std::vector<double> *>(ptr);
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
