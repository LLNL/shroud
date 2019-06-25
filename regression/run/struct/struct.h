/* Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC. 
 *
 * Produced at the Lawrence Livermore National Laboratory 
 *
 * LLNL-CODE-738041.
 *
 * All rights reserved. 
 *
 * This file is part of Shroud.
 *
 * For details about use and distribution, please read LICENSE.
 *
 * #######################################################################
 *
 * clibrary.hpp - wrapped routines
 */

#ifndef STRUCT_H
#define STRUCT_H

/* Size of buffer passed from Fortran */
#define LENOUTBUF 40

struct Cstruct1 {
  int ifield;
  double dfield;
};
typedef struct Cstruct1 Cstruct1;

int passStructByValue(Cstruct1 arg);
int passStruct1(Cstruct1 *s1);
int passStruct2(Cstruct1 *s1, char *outbuf);
int acceptStructInPtr(Cstruct1 *arg);
void acceptStructOutPtr(Cstruct1 *arg, int i, double d);
void acceptStructInOutPtr(Cstruct1 *arg);
Cstruct1 returnStruct(int i, double d);
Cstruct1 *returnStructPtr1(int i, double d);
Cstruct1 *returnStructPtr2(int i, double d, char *outbuf);

#endif // STRUCT_H
