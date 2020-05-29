! Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from tr291113.yaml.
!
program tester
  use fruit
  use iso_c_binding
  use tr29113_mod
  implicit none
  logical ok

  call init_fruit

  call test_tr29113

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_tr29113
    character(len=:), allocatable :: astr

    call set_case_name("test_tr29113")

    astr = get_const_string_ptr_alloc_tr()
    call assert_true( astr == "dog", "getConstStringPtrAlloc")

  end subroutine test_tr29113

end program tester
