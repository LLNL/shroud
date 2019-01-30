/*
 * Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC. 
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
 */
#include "Python.h"
//#include <pyTutorialmodule.hpp>
#include <stdio.h>

#if PY_MAJOR_VERSION >= 3
#define MODINIT  PyInit_tutorial
#else
#define MODINIT  inittutorial
#endif
PyMODINIT_FUNC MODINIT(void);

int main(int argc, char** argv)  
{
    char filename[] = "test.py";
    FILE* fp;

    Py_Initialize();
    MODINIT();
    
    fp = fopen(filename, "r");
    PyRun_SimpleFile(fp, filename);
    fclose(fp);
    Py_Exit(0);  
    return 0;
}
