! Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
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
  call test_indextype2

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

    arg1 = 20
    rv = typefunc_wrap(arg1)
    call assert_equals(rv, arg1 + 1, "typefunc_wrap")

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
    integer(INDEX_TYPE) shapearg(2), sizerv
    
    call set_case_name("test_index")

    arg = 0_INDEX_TYPE
    nbytes = return_bytes_for_index_type(arg)
#if defined(USE_64BIT_INDEXTYPE)
    call assert_equals(8, nbytes, "returnBytesForIndexType")
#else
    call assert_equals(4, nbytes, "returnBytesForIndexType")
#endif

    shapearg = [2, 3]
    sizerv = return_shape_size(size(shapearg), shapearg)
    call assert_equals(6, sizerv, "returnShapeSize2")
    
  end subroutine test_indextype

  subroutine test_indextype2
    integer nbytes
    integer(LOCAL_INDEX_TYPE) arg
    integer(LOCAL_INDEX_TYPE) shapearg(2), sizerv
    
    call set_case_name("test_index2")

    arg = 0_LOCAL_INDEX_TYPE
    nbytes = return_bytes_for_index_type2(arg)
#if defined(USE_64BIT_INDEXTYPE)
    call assert_equals(8, nbytes, "returnBytesForIndexType2")
#else
    call assert_equals(4, nbytes, "returnBytesForIndexType2")
#endif

    shapearg = [2, 3]
    sizerv = return_shape_size2(size(shapearg), shapearg)
    call assert_equals(6, sizerv, "returnShapeSize2")
    
  end subroutine test_indextype2

end program tester

