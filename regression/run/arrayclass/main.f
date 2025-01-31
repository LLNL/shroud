! Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from arrayclass.yaml.
!
program tester
  use fruit
  use iso_c_binding
  use arrayclass_mod
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

!   generic calls ArrayWrapper_ctor()
    arrinst = ArrayWrapper()
    call arrinst%setSize(10)
    call assert_equals(10, arrinst%getSize())

    call arrinst%fillSize(isize)
    call assert_equals(10, isize)

    call arrinst%allocate()
    arr => arrinst%getArray()
    call assert_true(associated(arr))
    call assert_equals(10, size(arr))

    ! Make sure we're pointing to the array in the instance.
    arr(:) = 0.0
    call assert_equals(0.0_C_DOUBLE, arrinst%sumArray())
    arr(:) = 1.0
    call assert_equals(10.0_C_DOUBLE, arrinst%sumArray())
    arr(:) = 0.0
    arr(1) = 10.0
    arr(10) = 1.0
    call assert_equals(11.0_C_DOUBLE, arrinst%sumarray())

    arrconst => arrinst%getArrayConst()
    call assert_true(associated(arrconst, arr))
    call assert_equals(10, size(arrconst))

    arr3 => arrinst%getArrayC()
    call assert_true(associated(arr3, arr))
    call assert_equals(10, size(arr3))

    arr4 => arrinst%getArrayConstC()
    call assert_true(associated(arr4, arr))
    call assert_equals(10, size(arr4))

    call arrinst%fetchArrayPtr(arr5)
    call assert_true(associated(arr5, arr))
    call assert_equals(10, size(arr5))

    call arrinst%fetchArrayRef(arr6)
    call assert_true(associated(arr6, arr))
    call assert_equals(10, size(arr6))

    call arrinst%fetchArrayPtrConst(arr7)
    call assert_true(associated(arr7, arr))
    call assert_equals(10, size(arr7))

    call arrinst%fetchArrayRefConst(arr8)
    call assert_true(associated(arr8, arr))
    call assert_equals(10, size(arr8))

    voidptr = C_NULL_PTR
    call arrinst%fetchVoidPtr(voidptr)
    call assert_true(c_associated(voidptr, c_loc(arr)), "fetchVoidPtr")
    call assert_true(arrinst%checkPtr(voidptr), "checkPtr")

    voidptr = C_NULL_PTR
    call arrinst%fetchVoidRef(voidptr)
    call assert_true(c_associated(voidptr, c_loc(arr)), "fetchVoidRef")
    call assert_true(arrinst%checkPtr(voidptr), "checkPtr")

  end subroutine test_arraywrapper
    
end program tester
