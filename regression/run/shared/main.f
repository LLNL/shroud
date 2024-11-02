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
  call test_object_shared

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_object
    type(object) objectPtr

    call set_case_name("test_object")

    objectPtr = object()
    call assert_true(objectPtr%associated())

  end subroutine test_object

  subroutine test_object_shared
    type(object_shared) objectSharedPtr
    type(object_shared) childA, childB

    call set_case_name("test_object_shared")

    objectSharedPtr = object_shared()
    call assert_true(objectSharedPtr%associated())

    childA = objectSharedPtr%create_child_a()
    call assert_true(childA%associated())

    !    childB = objectSharedPtr%create_child_b()

  end subroutine test_object_shared

end program tester
