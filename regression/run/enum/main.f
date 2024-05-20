! Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from enum.yaml.
!
program tester
  use fruit
  use iso_c_binding
  use enum_mod
  implicit none
  logical ok

  call init_fruit

  call test_enums
  call test_enum_functions

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_enums
    ! test values of enumerations
    call set_case_name("test_enums")

    call assert_true(10 == red, "enum_color_red")
    call assert_true(11 == blue, "enum_color_blue")
    call assert_true(12 == white, "enum_color_white")

    call assert_equals(0, a1, "enum_val_a1")
    call assert_equals(3, b1, "enum_val_b1")
    call assert_equals(4, c1, "enum_val_c1")
    call assert_equals(3, d1, "enum_val_d1")
    call assert_equals(3, e1, "enum_val_e1")
    call assert_equals(4, f1, "enum_val_f1")
    call assert_equals(5, g1, "enum_val_g1")
    call assert_equals(100, h1, "enum_val_h1")

  end subroutine test_enums

  subroutine test_enum_functions
    ! test functions which pass enums

    integer icol
    integer(C_SHORT) outcolor

    call set_case_name("test_enum_functions")

    icol = convert_to_int(RED)
    call assert_true(RED == icol, "convert_to_int")

    outcolor = return_enum(RED)
    call assert_true(RED == outcolor, "returnEnum")

    call return_enum_out_arg(outcolor)
    call assert_true(BLUE == outcolor, "returnEnumAsArg")
    
  end subroutine test_enum_functions
  
end program tester
