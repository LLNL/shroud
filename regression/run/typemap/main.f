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
    integer :: indx
#if defined(USE_64BIT_INDEXTYPE)
    integer(C_INT64_t) :: indx64
#else
    integer(C_INT32_t) :: indx32
#endif

    indx = 0
    call pass_index(indx)

#if defined(USE_64BIT_INDEXTYPE)
    indx64 = 0
!    call pass_index(indx64)
#else
    indx32 = 0
    call pass_index(indx32)
#endif

  end subroutine test_indextype
end program tester
  
