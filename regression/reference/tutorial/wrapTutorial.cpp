// wrapTutorial.cpp
// This is generated code, do not edit
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
#include "wrapTutorial.h"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include "tutorial.hpp"
#include "typesTutorial.h"

// splicer begin CXX_definitions
// splicer end CXX_definitions

extern "C" {


// helper function
// Copy src into dest, blank fill to ndest characters
// Truncate if dest is too short.
// dest will not be NULL terminated.
static void ShroudStrCopy(char *dest, int ndest, const char *src, int nsrc)
{
   if (src == NULL) {
     std::memset(dest,' ',ndest); // convert NULL pointer to blank filled string
   } else {
     if (nsrc < 0) nsrc = std::strlen(src);
     int nm = nsrc < ndest ? nsrc : ndest;
     std::memcpy(dest,src,nm);
     if(ndest > nm) std::memset(dest+nm,' ',ndest-nm); // blank fill
   }
}

// helper function
// start helper ShroudStrToArray
// Save str metadata into array to allow Fortran to access values.
static void ShroudStrToArray(TUT_SHROUD_array *array, const std::string * src, int idtor)
{
    array->cxx.addr = static_cast<void *>(const_cast<std::string *>(src));
    array->cxx.idtor = idtor;
    if (src->empty()) {
        array->addr.ccharp = NULL;
        array->len = 0;
    } else {
        array->addr.ccharp = src->data();
        array->len = src->size();
    }
    array->size = 1;
}
// end helper ShroudStrToArray

// helper function
// start helper copy_string
// Copy the char* or std::string in context into c_var.
// Called by Fortran to deal with allocatable character.
void TUT_ShroudCopyStringAndFree(TUT_SHROUD_array *data, char *c_var, size_t c_var_len) {
    const char *cxx_var = data->addr.ccharp;
    size_t n = c_var_len;
    if (data->len < n) n = data->len;
    std::strncpy(c_var, cxx_var, n);
    TUT_SHROUD_memory_destructor(&data->cxx); // delete data->cxx.addr
}
// end helper copy_string

// splicer begin C_definitions
// splicer end C_definitions

// void NoReturnNoArguments()
// start TUT_no_return_no_arguments
void TUT_no_return_no_arguments()
{
// splicer begin function.no_return_no_arguments
    tutorial::NoReturnNoArguments();
    return;
// splicer end function.no_return_no_arguments
}
// end TUT_no_return_no_arguments

// double PassByValue(double arg1 +intent(in)+value, int arg2 +intent(in)+value)
double TUT_pass_by_value(double arg1, int arg2)
{
// splicer begin function.pass_by_value
    double SHC_rv = tutorial::PassByValue(arg1, arg2);
    return SHC_rv;
// splicer end function.pass_by_value
}

// void ConcatenateStrings(const std::string & arg1 +intent(in)+len_trim(Larg1), const std::string & arg2 +intent(in)+len_trim(Larg2), const std::string * SHF_rv +context(DSHF_rv)+deref(allocatable)+intent(out))
/**
 * Note that since a reference is returned, no intermediate string
 * is allocated.  It is assumed +owner(library).
 */
void TUT_concatenate_strings_bufferify(const char * arg1, int Larg1,
    const char * arg2, int Larg2, TUT_SHROUD_array *DSHF_rv)
{
// splicer begin function.concatenate_strings_bufferify
    const std::string SHCXX_arg1(arg1, Larg1);
    const std::string SHCXX_arg2(arg2, Larg2);
    std::string * SHCXX_rv = new std::string;
    *SHCXX_rv = tutorial::ConcatenateStrings(SHCXX_arg1, SHCXX_arg2);
    ShroudStrToArray(DSHF_rv, SHCXX_rv, 2);
    return;
// splicer end function.concatenate_strings_bufferify
}

// double UseDefaultArguments()
// start TUT_use_default_arguments
double TUT_use_default_arguments()
{
// splicer begin function.use_default_arguments
    double SHC_rv = tutorial::UseDefaultArguments();
    return SHC_rv;
// splicer end function.use_default_arguments
}
// end TUT_use_default_arguments

// double UseDefaultArguments(double arg1=3.1415 +intent(in)+value)
// start TUT_use_default_arguments_arg1
double TUT_use_default_arguments_arg1(double arg1)
{
// splicer begin function.use_default_arguments_arg1
    double SHC_rv = tutorial::UseDefaultArguments(arg1);
    return SHC_rv;
// splicer end function.use_default_arguments_arg1
}
// end TUT_use_default_arguments_arg1

// double UseDefaultArguments(double arg1=3.1415 +intent(in)+value, bool arg2=true +intent(in)+value)
// start TUT_use_default_arguments_arg1_arg2
double TUT_use_default_arguments_arg1_arg2(double arg1, bool arg2)
{
// splicer begin function.use_default_arguments_arg1_arg2
    double SHC_rv = tutorial::UseDefaultArguments(arg1, arg2);
    return SHC_rv;
// splicer end function.use_default_arguments_arg1_arg2
}
// end TUT_use_default_arguments_arg1_arg2

// void OverloadedFunction(const std::string & name +intent(in))
void TUT_overloaded_function_from_name(const char * name)
{
// splicer begin function.overloaded_function_from_name
    const std::string SHCXX_name(name);
    tutorial::OverloadedFunction(SHCXX_name);
    return;
// splicer end function.overloaded_function_from_name
}

// void OverloadedFunction(const std::string & name +intent(in)+len_trim(Lname))
void TUT_overloaded_function_from_name_bufferify(const char * name,
    int Lname)
{
// splicer begin function.overloaded_function_from_name_bufferify
    const std::string SHCXX_name(name, Lname);
    tutorial::OverloadedFunction(SHCXX_name);
    return;
// splicer end function.overloaded_function_from_name_bufferify
}

// void OverloadedFunction(int indx +intent(in)+value)
void TUT_overloaded_function_from_index(int indx)
{
// splicer begin function.overloaded_function_from_index
    tutorial::OverloadedFunction(indx);
    return;
// splicer end function.overloaded_function_from_index
}

// void TemplateArgument(int arg +intent(in)+value)
void TUT_template_argument_int(int arg)
{
// splicer begin function.template_argument_int
    tutorial::TemplateArgument<int>(arg);
    return;
// splicer end function.template_argument_int
}

// void TemplateArgument(double arg +intent(in)+value)
void TUT_template_argument_double(double arg)
{
// splicer begin function.template_argument_double
    tutorial::TemplateArgument<double>(arg);
    return;
// splicer end function.template_argument_double
}

// int TemplateReturn()
int TUT_template_return_int()
{
// splicer begin function.template_return_int
    int SHC_rv = tutorial::TemplateReturn<int>();
    return SHC_rv;
// splicer end function.template_return_int
}

// double TemplateReturn()
double TUT_template_return_double()
{
// splicer begin function.template_return_double
    double SHC_rv = tutorial::TemplateReturn<double>();
    return SHC_rv;
// splicer end function.template_return_double
}

// void FortranGenericOverloaded()
void TUT_fortran_generic_overloaded_0()
{
// splicer begin function.fortran_generic_overloaded_0
    tutorial::FortranGenericOverloaded();
    return;
// splicer end function.fortran_generic_overloaded_0
}

// void FortranGenericOverloaded(const std::string & name +intent(in), double arg2 +intent(in)+value)
void TUT_fortran_generic_overloaded_1(const char * name, double arg2)
{
// splicer begin function.fortran_generic_overloaded_1
    const std::string SHCXX_name(name);
    tutorial::FortranGenericOverloaded(SHCXX_name, arg2);
    return;
// splicer end function.fortran_generic_overloaded_1
}

// void FortranGenericOverloaded(const std::string & name +intent(in)+len_trim(Lname), double arg2 +intent(in)+value)
void TUT_fortran_generic_overloaded_1_bufferify(const char * name,
    int Lname, double arg2)
{
// splicer begin function.fortran_generic_overloaded_1_bufferify
    const std::string SHCXX_name(name, Lname);
    tutorial::FortranGenericOverloaded(SHCXX_name, arg2);
    return;
// splicer end function.fortran_generic_overloaded_1_bufferify
}

// int UseDefaultOverload(int num +intent(in)+value)
int TUT_use_default_overload_num(int num)
{
// splicer begin function.use_default_overload_num
    int SHC_rv = tutorial::UseDefaultOverload(num);
    return SHC_rv;
// splicer end function.use_default_overload_num
}

// int UseDefaultOverload(int num +intent(in)+value, int offset=0 +intent(in)+value)
int TUT_use_default_overload_num_offset(int num, int offset)
{
// splicer begin function.use_default_overload_num_offset
    int SHC_rv = tutorial::UseDefaultOverload(num, offset);
    return SHC_rv;
// splicer end function.use_default_overload_num_offset
}

// int UseDefaultOverload(int num +intent(in)+value, int offset=0 +intent(in)+value, int stride=1 +intent(in)+value)
int TUT_use_default_overload_num_offset_stride(int num, int offset,
    int stride)
{
// splicer begin function.use_default_overload_num_offset_stride
    int SHC_rv = tutorial::UseDefaultOverload(num, offset, stride);
    return SHC_rv;
// splicer end function.use_default_overload_num_offset_stride
}

// int UseDefaultOverload(double type +intent(in)+value, int num +intent(in)+value)
int TUT_use_default_overload_3(double type, int num)
{
// splicer begin function.use_default_overload_3
    int SHC_rv = tutorial::UseDefaultOverload(type, num);
    return SHC_rv;
// splicer end function.use_default_overload_3
}

// int UseDefaultOverload(double type +intent(in)+value, int num +intent(in)+value, int offset=0 +intent(in)+value)
int TUT_use_default_overload_4(double type, int num, int offset)
{
// splicer begin function.use_default_overload_4
    int SHC_rv = tutorial::UseDefaultOverload(type, num, offset);
    return SHC_rv;
// splicer end function.use_default_overload_4
}

// int UseDefaultOverload(double type +intent(in)+value, int num +intent(in)+value, int offset=0 +intent(in)+value, int stride=1 +intent(in)+value)
int TUT_use_default_overload_5(double type, int num, int offset,
    int stride)
{
// splicer begin function.use_default_overload_5
    int SHC_rv = tutorial::UseDefaultOverload(type, num, offset,
        stride);
    return SHC_rv;
// splicer end function.use_default_overload_5
}

// TypeID typefunc(TypeID arg +intent(in)+value)
int TUT_typefunc(int arg)
{
// splicer begin function.typefunc
    tutorial::TypeID SHC_rv = tutorial::typefunc(arg);
    return SHC_rv;
// splicer end function.typefunc
}

// EnumTypeID enumfunc(EnumTypeID arg +intent(in)+value)
int TUT_enumfunc(int arg)
{
// splicer begin function.enumfunc
    tutorial::EnumTypeID SHCXX_arg =
        static_cast<tutorial::EnumTypeID>(arg);
    tutorial::EnumTypeID SHCXX_rv = tutorial::enumfunc(SHCXX_arg);
    int SHC_rv = static_cast<int>(SHCXX_rv);
    return SHC_rv;
// splicer end function.enumfunc
}

// Color colorfunc(Color arg +intent(in)+value)
int TUT_colorfunc(int arg)
{
// splicer begin function.colorfunc
    tutorial::Color SHCXX_arg = static_cast<tutorial::Color>(arg);
    tutorial::Color SHCXX_rv = tutorial::colorfunc(SHCXX_arg);
    int SHC_rv = static_cast<int>(SHCXX_rv);
    return SHC_rv;
// splicer end function.colorfunc
}

// void getMinMax(int & min +intent(out), int & max +intent(out))
/**
 * \brief Pass in reference to scalar
 *
 */
// start TUT_get_min_max
void TUT_get_min_max(int * min, int * max)
{
// splicer begin function.get_min_max
    tutorial::getMinMax(*min, *max);
    return;
// splicer end function.get_min_max
}
// end TUT_get_min_max

// Class1::DIRECTION directionFunc(Class1::DIRECTION arg +intent(in)+value)
int TUT_direction_func(int arg)
{
// splicer begin function.direction_func
    tutorial::Class1::DIRECTION SHCXX_arg =
        static_cast<tutorial::Class1::DIRECTION>(arg);
    tutorial::Class1::DIRECTION SHCXX_rv = tutorial::directionFunc(
        SHCXX_arg);
    int SHC_rv = static_cast<int>(SHCXX_rv);
    return SHC_rv;
// splicer end function.direction_func
}

// void passClassByValue(Class1 arg +intent(in)+value)
/**
 * \brief Pass arguments to a function.
 *
 */
void TUT_pass_class_by_value(TUT_Class1 * arg)
{
// splicer begin function.pass_class_by_value
    tutorial::Class1 * SHCXX_arg =
        static_cast<tutorial::Class1 *>(arg->addr);
    tutorial::passClassByValue(*SHCXX_arg);
    return;
// splicer end function.pass_class_by_value
}

// int useclass(const Class1 * arg +intent(in))
int TUT_useclass(TUT_Class1 * arg)
{
// splicer begin function.useclass
    const tutorial::Class1 * SHCXX_arg =
        static_cast<const tutorial::Class1 *>(arg->addr);
    int SHC_rv = tutorial::useclass(SHCXX_arg);
    return SHC_rv;
// splicer end function.useclass
}

// const Class1 * getclass2()
TUT_Class1 * TUT_getclass2(TUT_Class1 * SHC_rv)
{
// splicer begin function.getclass2
    const tutorial::Class1 * SHCXX_rv = tutorial::getclass2();
    SHC_rv->addr = static_cast<void *>(const_cast<tutorial::Class1 *>
        (SHCXX_rv));
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end function.getclass2
}

// Class1 * getclass3()
TUT_Class1 * TUT_getclass3(TUT_Class1 * SHC_rv)
{
// splicer begin function.getclass3
    tutorial::Class1 * SHCXX_rv = tutorial::getclass3();
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 0;
    return SHC_rv;
// splicer end function.getclass3
}

// Class1 getClassCopy(int flag +intent(in)+value)
/**
 * \brief Return Class1 instance by value, uses copy constructor
 *
 */
TUT_Class1 * TUT_get_class_copy(int flag, TUT_Class1 * SHC_rv)
{
// splicer begin function.get_class_copy
    tutorial::Class1 * SHCXX_rv = new tutorial::Class1;
    *SHCXX_rv = tutorial::getClassCopy(flag);
    SHC_rv->addr = static_cast<void *>(SHCXX_rv);
    SHC_rv->idtor = 1;
    return SHC_rv;
// splicer end function.get_class_copy
}

// int callback1(int in +intent(in)+value, int ( * incr)(int +value) +intent(in)+value)
/**
 * \brief Test function pointer
 *
 */
// start TUT_callback1
int TUT_callback1(int in, int ( * incr)(int))
{
// splicer begin function.callback1
    int SHC_rv = tutorial::callback1(in, incr);
    return SHC_rv;
// splicer end function.callback1
}
// end TUT_callback1

// void set_global_flag(int arg +intent(in)+value)
void TUT_set_global_flag(int arg)
{
// splicer begin function.set_global_flag
    tutorial::set_global_flag(arg);
    return;
// splicer end function.set_global_flag
}

// int get_global_flag()
int TUT_get_global_flag()
{
// splicer begin function.get_global_flag
    int SHC_rv = tutorial::get_global_flag();
    return SHC_rv;
// splicer end function.get_global_flag
}

// const std::string & LastFunctionCalled() +deref(result_as_arg)+len(30)
const char * TUT_last_function_called()
{
// splicer begin function.last_function_called
    const std::string & SHCXX_rv = tutorial::LastFunctionCalled();
    const char * SHC_rv = SHCXX_rv.c_str();
    return SHC_rv;
// splicer end function.last_function_called
}

// void LastFunctionCalled(std::string & SHF_rv +intent(out)+len(NSHF_rv)) +len(30)
void TUT_last_function_called_bufferify(char * SHF_rv, int NSHF_rv)
{
// splicer begin function.last_function_called_bufferify
    const std::string & SHCXX_rv = tutorial::LastFunctionCalled();
    if (SHCXX_rv.empty()) {
        ShroudStrCopy(SHF_rv, NSHF_rv, NULL, 0);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHCXX_rv.data(),
            SHCXX_rv.size());
    }
    return;
// splicer end function.last_function_called_bufferify
}

// start release allocated memory
// Release library allocated memory.
void TUT_SHROUD_memory_destructor(TUT_SHROUD_capsule_data *cap)
{
    void *ptr = cap->addr;
    switch (cap->idtor) {
    case 0:   // --none--
    {
        // Nothing to delete
        break;
    }
    case 1:   // tutorial::Class1
    {
        tutorial::Class1 *cxx_ptr = 
            reinterpret_cast<tutorial::Class1 *>(ptr);
        delete cxx_ptr;
        break;
    }
    case 2:   // new_string
    {
        std::string *cxx_ptr = reinterpret_cast<std::string *>(ptr);
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
// end release allocated memory

}  // extern "C"
