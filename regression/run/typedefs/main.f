! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from typedefs.yaml.
! Used by typedefs-c, typedefs-cxx
!
program tester
  use fruit
  use iso_c_binding
  use typedefs_mod
  implicit none
  logical ok

  integer, parameter :: lenoutbuf = 40

  call init_fruit

  call test_alias

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_alias

    integer(Type_ID) arg1, rv

    arg1 = 10
    rv = typefunc(arg1)
    call assert_equals(rv, arg1 + 1, "typefunc")

  end subroutine test_alias
end program tester

