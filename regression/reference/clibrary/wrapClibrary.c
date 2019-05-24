// wrapClibrary.c
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738041.
//
// All rights reserved.
//
// This file is part of Shroud.
//
// For details about use and distribution, please read LICENSE.
//
// #######################################################################
#include "wrapClibrary.h"
#include <stdlib.h>
#include <string.h>
#include "clibrary.h"
#include "typesClibrary.h"


// helper function
// Copy src into new memory and null terminate.
static char *ShroudStrAlloc(const char *src, int nsrc, int ntrim)
{
   char *rv = malloc(nsrc + 1);
   if (ntrim > 0) {
     memcpy(rv, src, ntrim);
   }
   rv[ntrim] = '\0';
   return rv;
}

// helper function
// Copy src into dest, blank fill to ndest characters
// Truncate if dest is too short.
// dest will not be NULL terminated.
static void ShroudStrCopy(char *dest, int ndest, const char *src, int nsrc)
{
   if (src == NULL) {
     memset(dest,' ',ndest); // convert NULL pointer to blank filled string
   } else {
     if (nsrc < 0) nsrc = strlen(src);
     int nm = nsrc < ndest ? nsrc : ndest;
     memcpy(dest,src,nm);
     if(ndest > nm) memset(dest+nm,' ',ndest-nm); // blank fill
   }
}

// helper function
// Release memory allocated by ShroudStrAlloc
static void ShroudStrFree(char *src)
{
   free(src);
}
// splicer begin C_definitions
// splicer end C_definitions

// void Function4a(const char * arg1 +intent(in)+len_trim(Larg1), const char * arg2 +intent(in)+len_trim(Larg2), char * SHF_rv +intent(out)+len(NSHF_rv)) +len(30)
void CLI_function4a_bufferify(const char * arg1, int Larg1,
    const char * arg2, int Larg2, char * SHF_rv, int NSHF_rv)
{
// splicer begin function.function4a_bufferify
    char * SH_arg1 = ShroudStrAlloc(arg1, Larg1, Larg1);
    char * SH_arg2 = ShroudStrAlloc(arg2, Larg2, Larg2);
    char * SHC_rv = Function4a(SH_arg1, SH_arg2);
    ShroudStrFree(SH_arg1);
    ShroudStrFree(SH_arg2);
    ShroudStrCopy(SHF_rv, NSHF_rv, SHC_rv, -1);
    return;
// splicer end function.function4a_bufferify
}

// void returnOneName(char * name1 +charlen(MAXNAME)+intent(out)+len(Nname1))
/**
 * \brief Test charlen attribute
 *
 * Each argument is assumed to be at least MAXNAME long.
 * This define is provided by the user.
 * The function will copy into the user provided buffer.
 */
void CLI_return_one_name_bufferify(char * name1, int Nname1)
{
// splicer begin function.return_one_name_bufferify
    char * SH_name1 = ShroudStrAlloc(name1, Nname1, 0);
    returnOneName(SH_name1);
    ShroudStrCopy(name1, Nname1, SH_name1, -1);
    ShroudStrFree(SH_name1);
    return;
// splicer end function.return_one_name_bufferify
}

// void returnTwoNames(char * name1 +charlen(MAXNAME)+intent(out)+len(Nname1), char * name2 +charlen(MAXNAME)+intent(out)+len(Nname2))
/**
 * \brief Test charlen attribute
 *
 * Each argument is assumed to be at least MAXNAME long.
 * This define is provided by the user.
 * The function will copy into the user provided buffer.
 */
void CLI_return_two_names_bufferify(char * name1, int Nname1,
    char * name2, int Nname2)
{
// splicer begin function.return_two_names_bufferify
    char * SH_name1 = ShroudStrAlloc(name1, Nname1, 0);
    char * SH_name2 = ShroudStrAlloc(name2, Nname2, 0);
    returnTwoNames(SH_name1, SH_name2);
    ShroudStrCopy(name1, Nname1, SH_name1, -1);
    ShroudStrFree(SH_name1);
    ShroudStrCopy(name2, Nname2, SH_name2, -1);
    ShroudStrFree(SH_name2);
    return;
// splicer end function.return_two_names_bufferify
}

// void bindC2(const char * name +intent(in)+len_trim(Lname))
/**
 * \brief Rename Fortran name for interface only function
 *
 * This creates a Fortran bufferify function and an interface.
 */
void CLI_bind_c2_bufferify(const char * name, int Lname)
{
// splicer begin function.bind_c2_bufferify
    char * SH_name = ShroudStrAlloc(name, Lname, Lname);
    bindC2(SH_name);
    ShroudStrFree(SH_name);
    return;
// splicer end function.bind_c2_bufferify
}

// int passAssumedTypeBuf(void * arg +assumedtype+intent(in), const char * name +intent(in)+len_trim(Lname))
/**
 * \brief Test assumed-type
 *
 * A bufferify function is created.
 * Should only be call with an C_INT argument, and will
 * return the value passed in.
 */
int CLI_pass_assumed_type_buf_bufferify(void * arg, const char * name,
    int Lname)
{
// splicer begin function.pass_assumed_type_buf_bufferify
    char * SH_name = ShroudStrAlloc(name, Lname, Lname);
    int SHC_rv = passAssumedTypeBuf(arg, SH_name);
    ShroudStrFree(SH_name);
    return SHC_rv;
// splicer end function.pass_assumed_type_buf_bufferify
}

// void callback3(const char * type +intent(in)+len_trim(Ltype), void * in +assumedtype+intent(in), void ( * incr)(int *) +external+intent(in)+value)
/**
 * \brief Test function pointer
 *
 * A bufferify function will be created.
 */
void CLI_callback3_bufferify(const char * type, int Ltype, void * in,
    void ( * incr)(int *))
{
// splicer begin function.callback3_bufferify
    char * SH_type = ShroudStrAlloc(type, Ltype, Ltype);
    callback3(SH_type, in, incr);
    ShroudStrFree(SH_type);
    return;
// splicer end function.callback3_bufferify
}

// int passStruct2(Cstruct1 * s1 +intent(in), const char * name +intent(in)+len_trim(Lname))
/**
 * Pass name argument which will build a bufferify function.
 */
int CLI_pass_struct2_bufferify(Cstruct1 * s1, const char * name,
    int Lname)
{
// splicer begin function.pass_struct2_bufferify
    char * SH_name = ShroudStrAlloc(name, Lname, Lname);
    int SHC_rv = passStruct2(s1, SH_name);
    ShroudStrFree(SH_name);
    return SHC_rv;
// splicer end function.pass_struct2_bufferify
}

// Cstruct1 * returnStructPtr2(int ifield +intent(in)+value, const char * name +intent(in)+len_trim(Lname))
/**
 * \brief Return a pointer to a struct
 *
 * Generates a bufferify C wrapper function.
 */
Cstruct1 * CLI_return_struct_ptr2_bufferify(int ifield,
    const char * name, int Lname)
{
// splicer begin function.return_struct_ptr2_bufferify
    char * SH_name = ShroudStrAlloc(name, Lname, Lname);
    Cstruct1 * SHC_rv = returnStructPtr2(ifield, SH_name);
    ShroudStrFree(SH_name);
    return SHC_rv;
// splicer end function.return_struct_ptr2_bufferify
}

// Release C++ allocated memory.
void CLI_SHROUD_memory_destructor(CLI_SHROUD_capsule_data *cap)
{
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}
