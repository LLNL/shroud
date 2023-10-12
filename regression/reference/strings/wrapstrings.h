// wrapstrings.h
// This file is generated by Shroud nowrite-version. Do not edit.
// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Shroud Project Developers.
// See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//
/**
 * \file wrapstrings.h
 * \brief Shroud generated wrapper for strings library
 */
// For C users and C++ implementation

#ifndef WRAPSTRINGS_H
#define WRAPSTRINGS_H

// shroud
#include "typesstrings.h"

// splicer begin CXX_declarations
// splicer end CXX_declarations

#ifdef __cplusplus
extern "C" {
#endif

// splicer begin C_declarations
// splicer end C_declarations

void STR_init_test(void);

void STR_passChar(char status);

void STR_passCharForce(char status);

void STR_returnChar(char *SHC_rv);

void STR_passCharPtr(char * dest, const char * src);

void STR_passCharPtr_bufferify(char *dest, int SHT_dest_len,
    const char * src);

void STR_passCharPtrInOut(char * s);

void STR_passCharPtrInOut_bufferify(char *s, int SHT_s_len);

const char * STR_getCharPtr1(void);

void STR_getCharPtr1_bufferify(STR_SHROUD_array *SHT_rv_cdesc);

const char * STR_getCharPtr2(void);

void STR_getCharPtr2_bufferify(char *SHC_rv, int SHT_rv_len);

const char * STR_getCharPtr3(void);

void STR_getCharPtr3_bufferify(char *output, int noutput);

const char * STR_getCharPtr4(void);

#ifdef HAVE_CHARACTER_POINTER_FUNCTION
const char * STR_getCharPtr5(void);
#endif

#ifdef HAVE_CHARACTER_POINTER_FUNCTION
void STR_getCharPtr5_bufferify(STR_SHROUD_array *SHT_rv_cdesc);
#endif

void STR_getConstStringResult_bufferify(STR_SHROUD_array *SHT_rv_cdesc);

void STR_getConstStringLen_bufferify(char *SHC_rv, int SHT_rv_len);

void STR_getConstStringAsArg_bufferify(char *output, int noutput);

void STR_getConstStringAlloc_bufferify(STR_SHROUD_array *SHT_rv_cdesc);

const char * STR_getConstStringRefPure(void);

void STR_getConstStringRefPure_bufferify(STR_SHROUD_array *SHT_rv_cdesc,
    STR_SHROUD_capsule_data *SHT_rv_capsule);

const char * STR_getConstStringRefLen(void);

void STR_getConstStringRefLen_bufferify(char *SHC_rv, int SHT_rv_len);

const char * STR_getConstStringRefAsArg(void);

void STR_getConstStringRefAsArg_bufferify(char *output, int noutput);

const char * STR_getConstStringRefLenEmpty(void);

void STR_getConstStringRefLenEmpty_bufferify(char *SHC_rv,
    int SHT_rv_len);

const char * STR_getConstStringRefAlloc(void);

void STR_getConstStringRefAlloc_bufferify(
    STR_SHROUD_array *SHT_rv_cdesc,
    STR_SHROUD_capsule_data *SHT_rv_capsule);

const char * STR_getConstStringPtrLen(void);

void STR_getConstStringPtrLen_bufferify(char *SHC_rv, int SHT_rv_len);

const char * STR_getConstStringPtrAlloc(void);

void STR_getConstStringPtrAlloc_bufferify(
    STR_SHROUD_array *SHT_rv_cdesc,
    STR_SHROUD_capsule_data *SHT_rv_capsule);

const char * STR_getConstStringPtrOwnsAlloc(void);

void STR_getConstStringPtrOwnsAlloc_bufferify(
    STR_SHROUD_array *SHT_rv_cdesc,
    STR_SHROUD_capsule_data *SHT_rv_capsule);

const char * STR_getConstStringPtrOwnsAllocPattern(void);

void STR_getConstStringPtrOwnsAllocPattern_bufferify(
    STR_SHROUD_array *SHT_rv_cdesc,
    STR_SHROUD_capsule_data *SHT_rv_capsule);

#ifdef HAVE_CHARACTER_POINTER_FUNCTION
const char * STR_getConstStringPtrPointer(void);
#endif

#ifdef HAVE_CHARACTER_POINTER_FUNCTION
void STR_getConstStringPtrPointer_bufferify(
    STR_SHROUD_array *SHT_rv_cdesc);
#endif

void STR_acceptStringConstReference(const char * arg1);

void STR_acceptStringConstReference_bufferify(char *arg1,
    int SHT_arg1_len);

void STR_acceptStringReferenceOut(char * arg1);

void STR_acceptStringReferenceOut_bufferify(char *arg1,
    int SHT_arg1_len);

void STR_acceptStringReference(char * arg1);

void STR_acceptStringReference_bufferify(char *arg1, int SHT_arg1_len);

void STR_acceptStringPointerConst(const char * arg1);

void STR_acceptStringPointerConst_bufferify(char *arg1,
    int SHT_arg1_len);

void STR_acceptStringPointer(char * arg1);

void STR_acceptStringPointer_bufferify(char *arg1, int SHT_arg1_len);

void STR_fetchStringPointer(char * arg1);

void STR_fetchStringPointer_bufferify(char *arg1, int SHT_arg1_len);

void STR_acceptStringPointerLen(char * arg1, int * nlen);

void STR_acceptStringPointerLen_bufferify(char *arg1, int SHT_arg1_len,
    int * nlen);

void STR_fetchStringPointerLen(char * arg1, int * nlen);

void STR_fetchStringPointerLen_bufferify(char *arg1, int SHT_arg1_len,
    int * nlen);

int STR_acceptStringInstance(char *arg1);

int STR_acceptStringInstance_bufferify(char *arg1, int SHT_arg1_len);

void STR_fetchArrayStringArg_bufferify(
    STR_SHROUD_array *SHT_strs_cdesc);

void STR_fetchArrayStringAlloc_bufferify(
    STR_SHROUD_array *SHT_strs_cdesc,
    STR_SHROUD_capsule_data *SHT_strs_capsule);

void STR_fetchArrayStringAllocLen_bufferify(
    STR_SHROUD_array *SHT_strs_cdesc,
    STR_SHROUD_capsule_data *SHT_strs_capsule);

void STR_explicit1(char * name);

void STR_explicit2(char * name);

void STR_explicit2_bufferify(char *name, int SHT_name_len);

void STR_CreturnChar(char *SHC_rv);

void STR_CpassCharPtr_bufferify(char *dest, int SHT_dest_len, char *src,
    int SHT_src_len);

void STR_CpassCharPtrBlank(char * dest, const char * src);

void STR_CpassCharPtrBlank_bufferify(char *dest, int SHT_dest_len,
    char *src, int SHT_src_len);

void STR_PostDeclare(int * count, char * name);

void STR_PostDeclare_bufferify(int * count, char *name,
    int SHT_name_len);

int STR_CpassCharPtrNotrim(const char * src);

int STR_CpassCharPtrNotrim_bufferify(char *src, int SHT_src_len);

int STR_CpassCharPtrCAPI(void * addr, const char * src);

int STR_CpassCharPtrCAPI2(const char * in, const char * src);

#ifdef __cplusplus
}
#endif

#endif  // WRAPSTRINGS_H
