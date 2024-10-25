! Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from shared.yaml.
!

program tester
  use fruit
  use iso_c_binding
  use shared_mod
  implicit none
  logical ok


  call init_fruit

  call test_object

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_object
    type(object) objectSharedPtr

    call set_case_name("test_object")

    objectSharedPtr = object()
    call assert_true(objectSharedPtr%associated())

  end subroutine test_object

end program tester
