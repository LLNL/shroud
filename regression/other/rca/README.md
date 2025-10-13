
Reference Counting Architecture
-------------------------------

Executable version of the code from [2].  Only the constructor/final
routines are implemented.  Not any of the vector operators.
It includes some debug functions to trace the flow.

The gfortran 13 included improvements to FINAL over gfortran 12 did
not call FINAL in the same sequence.
(The shared regression test passes with gcc12 but not gcc13. gcc13
releases the C++ object as part of the assignment due to the FINAL clause.)

OneAPI 2025.2.0 run the same as gfortran 13.

The xlf 2023.03.13 compiler has an issue witht the generic interface
for cpp_new_vector and requires that XLF be defined.

The Cray cce ftn 20.0.0 results in a Segmentation fault.


[1] Morris K., Rouson D. W. I., and Xia J., On the object-oriented
design of reference-counted shadow objects, Proceedings of the 4th
International Workshop on Software Engineering for Computational
Science and Engineering (SE-CSE ′11), May 2011, 19–27,

[2] Rouson D. W. I., Morris K., and Xia J., Managing C++ objects with
Fortran in the driver′s seat: this is not your parents′ Fortran,
Computing in Science and Engineering. (2012) 14, no. 2, 46–54.


```
make clean
make tester
./tester
```
