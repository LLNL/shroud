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
    integer(C_INT) isize
    type(ArrayWrapper) arrinst  ! instance
    real(C_DOUBLE), pointer :: arr(:), arrconst(:)
    real(C_DOUBLE), pointer :: arr3(:), arr4(:), arr5(:), arr6(:), &
         arr7(:), arr8(:)
    type(C_PTR) :: voidptr

    arrinst = ArrayWrapper_ctor()
    call arrinst%set_size(10)
    call assert_equals(10, arrinst%get_size())

    call arrinst%fill_size(isize)
    call assert_equals(10, isize)

    call arrinst%allocate()
    arr => arrinst%get_array()
    call assert_true(associated(arr))
    call assert_equals(10, size(arr))

    ! Make sure we're pointing to the array in the instance.
    arr(:) = 0.0
    call assert_equals(0.0_C_DOUBLE, arrinst%sum_array())
    arr(:) = 1.0
    call assert_equals(10.0_C_DOUBLE, arrinst%sum_array())
    arr(:) = 0.0
    arr(1) = 10.0
    arr(10) = 1.0
    call assert_equals(11.0_C_DOUBLE, arrinst%sum_array())

    arrconst => arrinst%get_array_const()
    call assert_true(associated(arrconst, arr))
    call assert_equals(10, size(arrconst))

    arr3 => arrinst%get_array_c()
    call assert_true(associated(arr3, arr))
    call assert_equals(10, size(arr3))

    arr4 => arrinst%get_array_const_c()
    call assert_true(associated(arr4, arr))
    call assert_equals(10, size(arr4))

    call arrinst%fetch_array_ptr(arr5)
    call assert_true(associated(arr5, arr))
    call assert_equals(10, size(arr5))

    call arrinst%fetch_array_ref(arr6)
    call assert_true(associated(arr6, arr))
    call assert_equals(10, size(arr6))

    call arrinst%fetch_array_ptr_const(arr7)
    call assert_true(associated(arr7, arr))
    call assert_equals(10, size(arr7))

    call arrinst%fetch_array_ref_const(arr8)
    call assert_true(associated(arr8, arr))
    call assert_equals(10, size(arr8))

    voidptr = C_NULL_PTR
    call arrinst%fetch_void_ptr(voidptr)
    call assert_true(c_associated(voidptr, c_loc(arr)), "fetchVoidPtr")
    call assert_true(arrinst%check_ptr(voidptr), "checkPtr")

    voidptr = C_NULL_PTR
    call arrinst%fetch_void_ref(voidptr)
    call assert_true(c_associated(voidptr, c_loc(arr)), "fetchVoidRef")
    call assert_true(arrinst%check_ptr(voidptr), "checkPtr")

  end subroutine test_arraywrapper
    
end program tester
