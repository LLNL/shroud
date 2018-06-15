// wrapClibrary.c
// This is generated code, do not edit
// #######################################################################
// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738041.
// All rights reserved.
//
// This file is part of Shroud.  For details, see
// https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the disclaimer (as noted below)
//   in the documentation and/or other materials provided with the
//   distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
// LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

// Release C++ allocated memory.
void CLI_SHROUD_memory_destructor(CLI_SHROUD_capsule_data *cap)
{
    cap->addr = NULL;
    cap->idtor = 0;  // avoid deleting again
}
