%module manual_mod

%{
#include <span>
%}

 // page 22
%include <typemaps.i>
%apply (SWIGTYPE *DATA, size_t SIZE) { (double *x, int x_length) };
%apply (const SWIGTYPE *DATA, size_t SIZE) { (const int *arr, size_t len) };
void fill_with_zeros(double* x, int x_length);
int accumulate(const int *arr, size_t len);

// page23
%include <std_span.i>
%template() std::span<int>;
std::span<int> get_array_ptr();
void set_array_ptr(std::span<int>& arr);
void increment(std::span<int> arr);
