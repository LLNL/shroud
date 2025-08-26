/*
 * Copyright Shroud Project Developers. See LICENSE file for details.
 *
 * SPDX-License-Identifier: (BSD-3-Clause)
 * #######################################################################
 */
#include "Python.h"
//#include <pyTutorialmodule.hpp>
#include <stdio.h>

#if PY_MAJOR_VERSION >= 3
#define MODINIT  PyInit_ccomplex
#else
#define MODINIT  initccomplex
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
