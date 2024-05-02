! Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from statement.yaml.
!
program tester
  use fruit
  use iso_c_binding
  use statement_mod
  implicit none
  logical ok

  call init_fruit

  call test_statement

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_statement
    ! test values of enumerations
    integer nlen
    character(20) name

    call set_case_name("test_statement")

    name = get_name_error_pattern()
    call assert_equals("the-name", name, "get_name_error_pattern")

    nlen = get_name_length()
    call assert_equals(len_trim(name), nlen  , "get_name_len")

    call assert_true(name_is_valid("dog"), "nameIsValid true")
    call assert_false(name_is_valid("  "), "nameIsValid false")
    
  end subroutine test_statement

end program tester
