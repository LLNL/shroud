/*
 * Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
 * other Shroud Project Developers.
 * See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 * #######################################################################
 *
 * struct.c
 */

#include "struct.h"
#include <stdlib.h>
#include <string.h>

#define MAXLAST 50
static char last_function_called[MAXLAST];

static Cstruct1 global_Cstruct1;

//----------------------------------------------------------------------

// return sum of fields as a check
// start passStructByValue
int passStructByValue(Cstruct1 arg)
{
  int rv = arg.ifield * 2;
  // Caller will not see changes.
  arg.ifield += 1;
  return rv;
}
// end passStructByValue

// start passStruct1
int passStruct1(const Cstruct1 *s1)
{
    strncpy(last_function_called, "passStruct1", MAXLAST);
    return s1->ifield;
}
// end passStruct1

int passStruct2(const Cstruct1 *s1, char *outbuf)
{
    strncpy(outbuf, "passStruct2", LENOUTBUF);
    strncpy(last_function_called, "passStruct2", MAXLAST);
    return s1->ifield;
}

// +intent(in)
// return sum of fields as a check
int acceptStructInPtr(Cstruct1 *arg)
{
  int rv = arg->ifield + arg->dfield;
  return rv;
}

// +intent(out)
void acceptStructOutPtr(Cstruct1 *arg, int i, double d)
{
  arg->ifield = i;
  arg->dfield = d;
  return;
}

// +intent(inout)
void acceptStructInOutPtr(Cstruct1 *arg)
{
  arg->ifield += 1;
  arg->dfield += 1.0;
  return;
}

Cstruct1 returnStructByValue(int i, double d)
{
  Cstruct1 s = {i, d};
  return s;
}

Cstruct1 *returnStructPtr1(int i, double d)
{
    strncpy(last_function_called, "returnStructPtr1", MAXLAST);
    global_Cstruct1.ifield = i;
    global_Cstruct1.dfield = d;
    return &global_Cstruct1;
}

Cstruct1 *returnStructPtr2(int i, double d, char *outbuf)
{
    strncpy(outbuf, "returnStructPtr2", LENOUTBUF);
    strncpy(last_function_called, "returnStructPtr2", MAXLAST);
    global_Cstruct1.ifield = i;
    global_Cstruct1.dfield = d;
    return &global_Cstruct1;
}

int callback1(Cstruct1 *arg, int (*work)(Cstruct1 *arg))
{
    int rv = work(arg);
    return rv;
}

#if 0
Cstruct1 *returnStructPtrNew(int i, double d)
{
  Cstruct1 *s = new Cstruct1;
  s->ifield = i;
  s->dfield = d;
  return s;
}

void freeStruct(Cstruct1 *arg1)
{
  delete arg1;
}
#endif

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

Cstruct_as_class *Create_Cstruct_as_class(void)
{
    Cstruct_as_class *rv = (Cstruct_as_class *)
        malloc(sizeof(Cstruct_as_class));
    rv->x1 = 0;
    rv->y1 = 0;
    return rv;
}

Cstruct_as_class *Create_Cstruct_as_class_args(int x, int y)
{
    Cstruct_as_class *rv = (Cstruct_as_class *)
        malloc(sizeof(Cstruct_as_class));
    rv->x1 = x;
    rv->y1 = y;
    return rv;
}

Cstruct_as_subclass *Create_Cstruct_as_subclass_args(int x, int y, int z)
{
    Cstruct_as_subclass *rv = (Cstruct_as_subclass *)
        malloc(sizeof(Cstruct_as_subclass));
    rv->x1 = x;
    rv->y1 = y;
    rv->z1 = z;
    return rv;
}

int Cstruct_as_class_sum(const Cstruct_as_class *point)
{
    return point->x1 + point->y1;
}

// Return sum of x members to test that they were passed in correctly.

int acceptBothStructs(Cstruct_as_class *s1, Cstruct_as_numpy *s2)
{
    int rv = s1->x1 + s2->x2;
    return rv;
}

/*----------------------------------------------------------------------*/
// Keep a global struct which owns all of its memory.
//   int *ivalue     +dimension(nitems+nitems);
//   double *dvalue  +dimension(nitems*TWO);
//   char **svalue   +dimension(nitems);

Cstruct_list *global_Cstruct_list = NULL;

Cstruct_list *get_global_struct_list(void)
{
    if (global_Cstruct_list == NULL) {
        int nitems = 4;

        int len = nitems + nitems;
        int *ivalue = (int *) malloc(sizeof(int) * len);
        for (int i=0; i < len; i++) {
            ivalue[i] = i;
        }

        len = nitems * TWO;
        double *dvalue = (double *) malloc(sizeof(double) * len);
        for (int i=0; i < len; i++) {
            dvalue[i] = i*2.0;
        }

        len = nitems;
        char **svalue = (char **) malloc(sizeof(char *) * len);
#ifdef __cplusplus
        // Avoid warning:
        // warning: deprecated conversion from string constant to 'char*'
        svalue[0] = (char *) malloc(3);
        strcpy(svalue[0], "up");
        svalue[1] = (char *) malloc(5);
        strcpy(svalue[1], "down");
        svalue[2] = (char *) malloc(5);
        strcpy(svalue[2], "left");
        svalue[3] = (char *) malloc(6);
        strcpy(svalue[3], "right");
#else
        svalue[0] = "up";
        svalue[1] = "down";
        svalue[2] = "left";
        svalue[3] = "right";
#endif

        Cstruct_list *work = (Cstruct_list *)
            malloc(sizeof(Cstruct_list));
        work->nitems = nitems;
        work->ivalue = ivalue;
        work->dvalue = dvalue;
        work->svalue = svalue;
        global_Cstruct_list = work;
    }
    return global_Cstruct_list;
}
