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
// Copy src into dest, blank fill to ndest characters
// Truncate if dest is too short.
// dest will not be NULL terminated.
static void ShroudStrCopy(char *dest, int ndest, const char *src, int nsrc)
{
   int nm = nsrc < ndest ? nsrc : ndest;
   memcpy(dest,src,nm);
   if(ndest > nm) memset(dest+nm,' ',ndest-nm);
}
// splicer begin C_definitions
// splicer end C_definitions

// void Function4a(const char * arg1 +intent(in)+len_trim(Larg1), const char * arg2 +intent(in)+len_trim(Larg2), char * SHF_rv +intent(out)+len(NSHF_rv)) +len(30)
void CLI_function4a_bufferify(const char * arg1, int Larg1,
    const char * arg2, int Larg2, char * SHF_rv, int NSHF_rv)
{
// splicer begin function.function4a_bufferify
    char * SH_arg1 = (char *) malloc(Larg1 + 1);
    memcpy(SH_arg1, arg1, Larg1);
    SH_arg1[Larg1] = '\0';
    char * SH_arg2 = (char *) malloc(Larg2 + 1);
    memcpy(SH_arg2, arg2, Larg2);
    SH_arg2[Larg2] = '\0';
    char * SHC_rv = Function4a(SH_arg1, SH_arg2);
    free(SH_arg1);
    free(SH_arg2);
    if (SHC_rv == NULL) {
        memset(SHF_rv, ' ', NSHF_rv);
    } else {
        ShroudStrCopy(SHF_rv, NSHF_rv, SHC_rv, strlen(SHC_rv));
    }
    return;
// splicer end function.function4a_bufferify
}

// void bindC2(const char * name +intent(in)+len_trim(Lname))
/**
 * \brief Rename Fortran name for interface only function
 *
 * This creates a Fortran implementation and an interface.
 */
void CLI_bind_c2_bufferify(const char * name, int Lname)
{
// splicer begin function.bind_c2_bufferify
    char * SH_name = (char *) malloc(Lname + 1);
    memcpy(SH_name, name, Lname);
    SH_name[Lname] = '\0';
    bindC2(SH_name);
    free(SH_name);
    return;
// splicer end function.bind_c2_bufferify
}

// Release C++ allocated memory.
void CLI_SHROUD_memory_destructor(CLI_SHROUD_capsule_data *cap)
{
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}
