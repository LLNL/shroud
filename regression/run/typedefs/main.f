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
  call test_struct
  call test_indextype

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_alias

    integer(Type_ID) arg1, rv

    call set_case_name("test_alias")

    arg1 = 10
    rv = typefunc(arg1)
    call assert_equals(rv, arg1 + 1, "typefunc")

  end subroutine test_alias

  subroutine test_struct
    type(struct1_rename) arg

    call set_case_name("test_struct")

    arg%i = 10
    arg%d = 0.0
    call typestruct(arg)
    call assert_equals(10._C_DOUBLE, arg%d, "typestruct")

  end subroutine test_struct

  subroutine test_indextype
    integer nbytes
    integer(INDEX_TYPE) arg
    
    call set_case_name("test_index")

    arg = 0_INDEX_TYPE
    nbytes = return_bytes_for_index_type(arg)
#if defined(USE_64BIT_INDEXTYPE)
    call assert_equals(8, nbytes, "return_bytes_for_index_type")
#else
    call assert_equals(4, nbytes, "return_bytes_for_index_type")
#endif
    
  end subroutine test_indextype
end program tester

