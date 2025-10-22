
## Test compiler final and assign interaction

```
make clean
make tester
./tester
```

The program creates some object and does assignment. It print out when
the `final` function is called.

gcc13 now calls `final` after the assignment overload. gcc12 did not.

Output from gcc 13 with annotations of differences from gcc 12.

```
 Called object_create: 1
 Called object_assign: none                =1
 Called object_final: 1                      <- missing from gcc12
 Called object_create: 2
 Called object_assign: none                =2
 Called object_final: 2                      <- missing from gcc12
 o1:assign 1
 o2:assign 2
 Test: object = object
 Called object_assign: assign 1            =assign 2
 Test: after assignment
 Called object_final: assign 2
 Called object_final: assign assign 2
```

Then intel 19 (classic ifort), intel 2025 (oneapi) and Cray ftn 20
also call the `final` subroutine as after the assignment.  But they
all reverse the order of the last two calls to `final`.

See also
https://fortran-lang.org/learn/oop_features_in_fortran/object_based_programming_techniques/
which suggests the rule: Finalizers, overloads for the default
constructor, and overload of the assignment operation should usually
be jointly implemented.
