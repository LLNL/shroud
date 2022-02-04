! Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from typemap.yaml.
!
program tester
  use fruit
  use iso_c_binding
  use typemap_mod
  implicit none
  logical ok

  call init_fruit

  call test_indextype

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_indextype
    integer(INDEXTYPE) :: indx
    integer(C_INT32_T) :: indx32
    integer(C_INT64_T) :: indx64

    indx = 0
!    call pass_index(indx)

    ! Match files with C.
    indx32 = 2
    indx64 = 2_C_INT64_T**34
#if defined(USE_64BIT_INDEXTYPE)
    call assert_equals(INDEXTYPE, C_INT64_T)
    call assert_false(pass_index(indx64 - 1, indx))
    call assert_true(pass_index(indx64, indx))
    call assert_true(indx == indx64)
#else
    call assert_equals(INDEXTYPE, C_INT32_T)
    call assert_false(pass_index(indx32 - 1, indx))
    call assert_true(pass_index(indx32, indx))
    call assert_true(indx == indx32)
#endif

#if defined(USE_64BIT_FLOAT)
    call assert_equals(FLOATTYPE, C_DOUBLE)
    call pass_float(1.0_C_DOUBLE)
#else
    call assert_equals(FLOATTYPE, C_FLOAT)
    call pass_float(1.0_C_FLOAT)
#endif

  end subroutine test_indextype
end program tester
  
