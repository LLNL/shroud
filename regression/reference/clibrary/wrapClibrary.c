// wrapClibrary.c
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//

// cxx_header
#include "clibrary.h"
// shroud
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "wrapClibrary.h"


// helper char_len_trim
// Returns the length of character string src with length nsrc,
// ignoring any trailing blanks.
static int ShroudCharLenTrim(const char *src, int nsrc) {
    int i;

    for (i = nsrc - 1; i >= 0; i--) {
        if (src[i] != ' ') {
            break;
        }
    }

    return i + 1;
}


// helper char_alloc
// Copy src into new memory and null terminate.
// If ntrim is 0, return NULL pointer.
// If blanknull is 1, return NULL when string is blank.
static char *ShroudCharAlloc(const char *src, int nsrc, int blanknull)
{
   int ntrim = ShroudCharLenTrim(src, nsrc);
   if (ntrim == 0 && blanknull == 1) {
     return NULL;
   }
   char *rv = malloc(nsrc + 1);
   if (ntrim > 0) {
     memcpy(rv, src, ntrim);
   }
   rv[ntrim] = '\0';
   return rv;
}

// helper char_blank_fill
// blank fill dest starting at trailing NULL.
static void ShroudCharBlankFill(char *dest, int ndest)
{
   int nm = strlen(dest);
   if(ndest > nm) memset(dest+nm,' ',ndest-nm);
}

// helper ShroudCharCopy
// Copy src into dest, blank fill to ndest characters
// Truncate if dest is too short.
// dest will not be NULL terminated.
static void ShroudCharCopy(char *dest, int ndest, const char *src, int nsrc)
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

// helper char_free
// Release memory allocated by ShroudCharAlloc
static void ShroudCharFree(char *src)
{
   if (src != NULL) {
     free(src);
   }
}
// splicer begin C_definitions
// splicer end C_definitions

/**
 * PassByValueMacro is a #define macro. Force a C wrapper
 * to allow Fortran to have an actual function to call.
 */
// ----------------------------------------
// Function:  double PassByValueMacro
// Attrs:     +intent(function)
// Statement: c_function_native_scalar
// ----------------------------------------
// Argument:  int arg2 +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
double CLI_PassByValueMacro(int arg2)
{
    // splicer begin function.PassByValueMacro
    double SHC_rv = PassByValueMacro(arg2);
    return SHC_rv;
    // splicer end function.PassByValueMacro
}

// Generated by arg_to_buffer
// ----------------------------------------
// Function:  char * Function4a +len(30)
// Attrs:     +api(buf)+deref(copy)+intent(function)
// Statement: f_function_char_*_buf_copy
// ----------------------------------------
// Argument:  const char * arg1
// Attrs:     +intent(in)
// Statement: f_in_char_*
// ----------------------------------------
// Argument:  const char * arg2
// Attrs:     +intent(in)
// Statement: f_in_char_*
void CLI_Function4a_bufferify(const char * arg1, const char * arg2,
    char *SHC_rv, int SHT_rv_len)
{
    // splicer begin function.Function4a_bufferify
    char * SHCXX_rv = Function4a(arg1, arg2);
    ShroudCharCopy(SHC_rv, SHT_rv_len, SHCXX_rv, -1);
    // splicer end function.Function4a_bufferify
}

/**
 * \brief toupper
 *
 * Change a string in-place.
 * For Python, return a new string since strings are immutable.
 */
// Generated by arg_to_buffer
// ----------------------------------------
// Function:  void passCharPtrInOut
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  char * s +intent(inout)
// Attrs:     +api(buf)+intent(inout)
// Statement: f_inout_char_*_buf
void CLI_passCharPtrInOut_bufferify(char *s, int SHT_s_len)
{
    // splicer begin function.passCharPtrInOut_bufferify
    char * SHT_s_str = ShroudCharAlloc(s, SHT_s_len, 0);
    passCharPtrInOut(SHT_s_str);
    ShroudCharCopy(s, SHT_s_len, SHT_s_str, -1);
    ShroudCharFree(SHT_s_str);
    // splicer end function.passCharPtrInOut_bufferify
}

/**
 * \brief Test charlen attribute
 *
 * Each argument is assumed to be at least MAXNAME long.
 * This define is provided by the user.
 * The function will copy into the user provided buffer.
 */
// Generated by arg_to_buffer
// ----------------------------------------
// Function:  void returnOneName
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  char * name1 +charlen(MAXNAME)+intent(out)
// Attrs:     +api(buf)+intent(out)
// Statement: f_out_char_*_buf
// start CLI_returnOneName_bufferify
void CLI_returnOneName_bufferify(char *name1, int SHT_name1_len)
{
    // splicer begin function.returnOneName_bufferify
    returnOneName(name1);
    ShroudCharBlankFill(name1, SHT_name1_len);
    // splicer end function.returnOneName_bufferify
}
// end CLI_returnOneName_bufferify

/**
 * \brief Test charlen attribute
 *
 * Each argument is assumed to be at least MAXNAME long.
 * This define is provided by the user.
 * The function will copy into the user provided buffer.
 */
// Generated by arg_to_buffer
// ----------------------------------------
// Function:  void returnTwoNames
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  char * name1 +charlen(MAXNAME)+intent(out)
// Attrs:     +api(buf)+intent(out)
// Statement: f_out_char_*_buf
// ----------------------------------------
// Argument:  char * name2 +charlen(MAXNAME)+intent(out)
// Attrs:     +api(buf)+intent(out)
// Statement: f_out_char_*_buf
void CLI_returnTwoNames_bufferify(char *name1, int SHT_name1_len,
    char *name2, int SHT_name2_len)
{
    // splicer begin function.returnTwoNames_bufferify
    returnTwoNames(name1, name2);
    ShroudCharBlankFill(name1, SHT_name1_len);
    ShroudCharBlankFill(name2, SHT_name2_len);
    // splicer end function.returnTwoNames_bufferify
}

/**
 * \brief Fill text, at most ltext characters.
 *
 */
// Generated by arg_to_buffer
// ----------------------------------------
// Function:  void ImpliedTextLen
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  char * text +charlen(MAXNAME)+intent(out)
// Attrs:     +api(buf)+intent(out)
// Statement: f_out_char_*_buf
// ----------------------------------------
// Argument:  int ltext +implied(len(text))+value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
// start CLI_ImpliedTextLen_bufferify
void CLI_ImpliedTextLen_bufferify(char *text, int SHT_text_len,
    int ltext)
{
    // splicer begin function.ImpliedTextLen_bufferify
    ImpliedTextLen(text, ltext);
    ShroudCharBlankFill(text, SHT_text_len);
    // splicer end function.ImpliedTextLen_bufferify
}
// end CLI_ImpliedTextLen_bufferify

/**
 * \brief Rename Fortran name for interface only function
 *
 * This creates a Fortran bufferify function and an interface.
 */
// Generated by arg_to_buffer
// ----------------------------------------
// Function:  void bindC2
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  char * outbuf +intent(out)
// Attrs:     +api(buf)+intent(out)
// Statement: f_out_char_*_buf
void CLI_bindC2_bufferify(char *outbuf, int SHT_outbuf_len)
{
    // splicer begin function.bindC2_bufferify
    bindC2(outbuf);
    ShroudCharBlankFill(outbuf, SHT_outbuf_len);
    // splicer end function.bindC2_bufferify
}

/**
 * \brief Test assumed-type
 *
 * A bufferify function is created.
 * Should only be call with an C_INT argument, and will
 * return the value passed in.
 */
// Generated by arg_to_buffer
// ----------------------------------------
// Function:  int passAssumedTypeBuf
// Attrs:     +intent(function)
// Statement: f_function_native_scalar
// ----------------------------------------
// Argument:  void * arg +assumedtype
// Attrs:     +intent(in)
// Statement: f_in_void_*
// ----------------------------------------
// Argument:  char * outbuf +intent(out)
// Attrs:     +api(buf)+intent(out)
// Statement: f_out_char_*_buf
int CLI_passAssumedTypeBuf_bufferify(void * arg, char *outbuf,
    int SHT_outbuf_len)
{
    // splicer begin function.passAssumedTypeBuf_bufferify
    int SHC_rv = passAssumedTypeBuf(arg, outbuf);
    ShroudCharBlankFill(outbuf, SHT_outbuf_len);
    return SHC_rv;
    // splicer end function.passAssumedTypeBuf_bufferify
}

/**
 * \brief Test function pointer
 *
 * Add C_force_wrapper to test generating function pointer prototype.
 */
// ----------------------------------------
// Function:  void callback1a
// Attrs:     +intent(subroutine)
// Statement: c_subroutine
// ----------------------------------------
// Argument:  int type +value
// Attrs:     +intent(in)
// Statement: f_in_native_scalar
// ----------------------------------------
// Argument:  void ( * incr)(void) +external+value
// Attrs:     +intent(in)
// Statement: f_in_void_scalar
// start CLI_callback1a
void CLI_callback1a(int type, void ( * incr)(void))
{
    // splicer begin function.callback1a
    callback1a(type, incr);
    // splicer end function.callback1a
}
// end CLI_callback1a

/**
 * \brief Test function pointer
 *
 * A bufferify function will be created.
 */
// Generated by arg_to_buffer
// ----------------------------------------
// Function:  void callback3
// Attrs:     +intent(subroutine)
// Statement: f_subroutine
// ----------------------------------------
// Argument:  const char * type
// Attrs:     +intent(in)
// Statement: f_in_char_*
// ----------------------------------------
// Argument:  void * in +assumedtype
// Attrs:     +intent(in)
// Statement: f_in_void_*
// ----------------------------------------
// Argument:  void ( * incr)(int *) +external+value
// Attrs:     +intent(in)
// Statement: f_in_void_scalar
// ----------------------------------------
// Argument:  char * outbuf +intent(out)
// Attrs:     +api(buf)+intent(out)
// Statement: f_out_char_*_buf
void CLI_callback3_bufferify(const char * type, void * in,
    void ( * incr)(int *), char *outbuf, int SHT_outbuf_len)
{
    // splicer begin function.callback3_bufferify
    callback3(type, in, incr, outbuf);
    ShroudCharBlankFill(outbuf, SHT_outbuf_len);
    // splicer end function.callback3_bufferify
}
