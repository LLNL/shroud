! Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from struct.yaml.
!
program tester
  use fruit
  use iso_c_binding
  use cxxlibrary_mod
  implicit none
  logical ok

  call init_fruit

  call test_struct

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_struct
    type(cstruct1) str1
    integer(C_INT) rvi

    call set_case_name("test_struct")

    str1 = cstruct1(2, 2.0)
    call assert_equals(2_C_INT, str1%ifield, "cstruct1 constructor ifield")
    call assert_equals(2.0_C_DOUBLE, str1%dfield, "cstruct1 constructor dfield")
    
    str1%dfield = 2.0_C_DOUBLE
    rvi = pass_struct_by_reference(str1)
    call assert_equals(4, rvi, "pass_struct_by_reference")
    ! Make sure str1 was passed by value.
!    call assert_equals(2_C_INT, str1%ifield, "pass_struct_by_value ifield")
!    call assert_equals(2.0_C_DOUBLE, str1%dfield, "pass_struct_by_value dfield")

  end subroutine test_struct

end program tester
