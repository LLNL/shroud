! Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from cdesc.yaml.
!
program tester
  use fruit
  use iso_c_binding
  use cdesc_mod
  implicit none
  logical ok

  call init_fruit

  call test_cdesc
  call test_cdesc2

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_cdesc
    ! test values of enumerations
    integer(C_INT) iarray2d(2,5)

    call set_case_name("test_cdesc")

    iarray2d = reshape([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 5])
    call rank2_in(iarray2d)

  end subroutine test_cdesc

  subroutine test_cdesc2
    ! test values of enumerations
    integer(C_INT) int_var
!    integer(C_LONG) long_var
!    real(C_FLOAT) float_var
    real(C_DOUBLE) double_var

    call set_case_name("test_cdesc2")

    int_var = 0
    call get_scalar1("name", int_var)
    call assert_equals(1, int_var, "get_scalar1 int")

!    long_var = 0
!    call get_scalar1("name", long_var)
!    call assert_equals(2, long_var, "get_scalar1 long")

!    float_var = 0
!    call get_scalar1("name", float_var)
!    call assert_equals(3.0, float_var, "get_scalar1 float")

    double_var = 0
    call get_scalar1("name", double_var)
    call assert_equals(4.0d0, double_var, "get_scalar1 double")

  end subroutine test_cdesc2

end program tester
