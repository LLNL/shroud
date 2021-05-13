
%module strings_mod
%{
#include "strings.hpp"
%}

#if 0
%include <typemaps.i>
%apply (SWIGTYPE *DATA, size_t SIZE) { (double *x, int x_length) };
%apply (const SWIGTYPE *DATA, size_t SIZE) { (const int *arr, size_t len) };

void fill_with_zeros(double* x, int x_length);
int accumulate(const int *arr, size_t len);
#endif

void passChar(char status);
char returnChar(void);
