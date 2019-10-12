// wrapns.cpp
// This is generated code, do not edit
#include "wrapns.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include "namespace.hpp"
#include "typesns.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


// helper function
// Copy the char* or std::string in context into c_var.
// Called by Fortran to deal with allocatable character.
void NS_ShroudCopyStringAndFree(NS_SHROUD_array *data, char *c_var, size_t c_var_len) {
    const char *cxx_var = data->addr.ccharp;
    size_t n = c_var_len;
    if (data->len < n) n = data->len;
    std::strncpy(c_var, cxx_var, n);
    NS_SHROUD_memory_destructor(&data->cxx); // delete data->cxx.addr
}

// splicer begin C_definitions
// splicer end C_definitions

// const std::string & LastFunctionCalled() +deref(allocatable)
const char * NS_last_function_called()
{
// splicer begin function.last_function_called
    const std::string & SHCXX_rv = LastFunctionCalled();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end function.last_function_called
}

// void LastFunctionCalled(const std::string * SHF_rv +context(DSHF_rv)+deref(allocatable)+intent(out))
void NS_last_function_called_bufferify(NS_SHROUD_array *DSHF_rv)
{
// splicer begin function.last_function_called_bufferify
    const std::string & SHCXX_rv = LastFunctionCalled();
    DSHF_rv->cxx.addr = static_cast<void *>(const_cast<std::string *>
        (&SHCXX_rv));
    DSHF_rv->cxx.idtor = 0;
    if (SHCXX_rv.empty()) {
        DSHF_rv->addr.ccharp = NULL;
        DSHF_rv->len = 0;
    } else {
        DSHF_rv->addr.ccharp = SHCXX_rv.data();
        DSHF_rv->len = SHCXX_rv.size();
    }
    DSHF_rv->size = 1;
    return;
// splicer end function.last_function_called_bufferify
}

// void One()
void NS_one()
{
// splicer begin function.one
    One();
    return;
// splicer end function.one
}

// Release library allocated memory.
void NS_SHROUD_memory_destructor(NS_SHROUD_capsule_data *cap)
{
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}

}  // extern "C"
