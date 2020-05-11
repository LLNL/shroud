! Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from references.yaml.
!
program tester
  use fruit
  use iso_c_binding
  use references_mod
  implicit none
  logical ok

  call init_fruit

  call test_arraywrapper

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_arraywrapper
    type(ArrayWrapper) arrinst  ! instance
    real(C_DOUBLE), pointer :: arr(:), arrconst(:)
    real(C_DOUBLE), pointer :: arr3(:), arr4(:)

    arrinst = ArrayWrapper_ctor()
    call arrinst%set_size(10)
    call assert_equals(10, arrinst%get_size())

    call arrinst%allocate()
    arr => arrinst%get_array()
    call assert_true(associated(arr))
    call assert_equals(10, size(arr))

    arrconst => arrinst%get_array_const()
    call assert_true(associated(arrconst, arr))
    call assert_equals(10, size(arrconst))

    arr3 => arrinst%get_array_c()
    call assert_true(associated(arrconst, arr))
    call assert_equals(10, size(arrconst))

    arr4 => arrinst%get_array_const_c()
    call assert_true(associated(arrconst, arr))
    call assert_equals(10, size(arrconst))
    
  end subroutine test_arraywrapper
    
end program tester
