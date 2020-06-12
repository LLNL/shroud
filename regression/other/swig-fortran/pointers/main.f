! Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from pointers.yaml.
! Used with pointers-c and pointers-cxx.
!
program tester
  use fruit
  use iso_c_binding
  use pointers_mod
  real(C_DOUBLE), parameter :: pi = 3.1415926_C_DOUBLE
  logical ok

  call init_fruit

  call test_swig

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_swig

    real(C_DOUBLE) zero(10)
    integer(C_INT) sum, count(5)

    call fill_with_zeros(zero)

    count = [1, 2, 3, 4, 5]
    sum = accumulate(count)
    call assert_equals(15, sum)
    
  end subroutine test_swig
  
end program tester
