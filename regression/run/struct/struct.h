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

// Used in dimension as a variable which is not a struct member.
#define TWO 2

struct Cstruct1 {
  int ifield;
  double dfield;
};
typedef struct Cstruct1 Cstruct1;

int passStructByValue(Cstruct1 arg);
int passStruct1(const Cstruct1 *s1);
int passStruct2(const Cstruct1 *s1, char *outbuf);
int acceptStructInPtr(Cstruct1 *arg);
void acceptStructOutPtr(Cstruct1 *arg, int i, double d);
void acceptStructInOutPtr(Cstruct1 *arg);
Cstruct1 returnStructByValue(int i, double d);
Cstruct1 *returnStructPtr1(int i, double d);
Cstruct1 *returnStructPtr2(int i, double d, char *outbuf);

int callback1(Cstruct1 *arg, int (*work)(Cstruct1 *arg));

/*----------------------------------------------------------------------*/
struct Cstruct_ptr {
    char *cfield;
    //    double *dvalue;              // ptr to scalar
    const double *const_dvalue;  // ptr to scalar
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
struct Arrays1 {
    char name[20];
    int count[10];
    //    int count[10][2];
};
typedef struct Arrays1 Arrays1;

/*----------------------------------------------------------------------*/
// Used in struct-py.yaml
// Test similar structs with PY_struct_arg as both "class" and "numpy"

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

Cstruct_as_class *Create_Cstruct_as_class(void);
Cstruct_as_class *Create_Cstruct_as_class_args(int x, int y);
int Cstruct_as_class_sum(const Cstruct_as_class *point);

int acceptBothStructs(Cstruct_as_class *s1, Cstruct_as_numpy *s2);

Cstruct_list *get_global_struct_list(void);

#endif // STRUCT_H
