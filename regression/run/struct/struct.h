/* This is generated code, do not edit
 * Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 *
 * struct.h - wrapped routines
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
Cstruct1 returnStructByValue(int i, double d);
const Cstruct1 returnConstStructByValue(int i, double d);
Cstruct1 *returnStructPtr1(int i, double d);
Cstruct1 *returnStructPtr2(int i, double d, char *outbuf);

/*----------------------------------------------------------------------*/
struct Cstruct_ptr {
    char *cfield;
};
typedef struct Cstruct_ptr Cstruct_ptr;

/*----------------------------------------------------------------------*/
struct Cstruct_list {
    int nitems;
    int *ivalue;
    double *dvalue;
    char **svalue;
};
typedef struct Cstruct_list Cstruct_list;

/*----------------------------------------------------------------------*/
struct Cstruct_numpy {
    int nitems;
    int *ivalue;
    double *dvalue;
};
typedef struct Cstruct_numpy Cstruct_numpy;

/*----------------------------------------------------------------------*/

struct Cstruct_as_class {
    int x1;
    int y1;
};
typedef struct Cstruct_as_class Cstruct_as_class;

struct Cstruct_as_numpy {
    int x2;
    int y2;
};
typedef struct Cstruct_as_numpy Cstruct_as_numpy;

int acceptBothStructs(Cstruct_as_class *s1, Cstruct_as_numpy *s2);


#endif // STRUCT_H
