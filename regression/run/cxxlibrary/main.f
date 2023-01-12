! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from struct.yaml.
!
program tester
  use fruit
  use iso_c_binding
  implicit none
  logical ok

  call init_fruit

  call test_struct
  call test_default_args
  call test_generic

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_struct
    use cxxlibrary_structns_mod
    type(cstruct1) str1, str2
    integer(C_INT) rvi

    call set_case_name("test_struct")

    str1 = cstruct1(2, 2.0)
    call assert_equals(2_C_INT, str1%ifield, "cstruct1 constructor ifield")
    call assert_equals(2.0_C_DOUBLE, str1%dfield, "cstruct1 constructor dfield")
    
    str1%dfield = 2.0_C_DOUBLE
    rvi = pass_struct_by_reference(str1)
    call assert_equals(4, rvi, "passStructByReference")
    ! Make sure str1 was passed by value.
!    call assert_equals(2_C_INT, str1%ifield, "pass_struct_by_value ifield")
!    call assert_equals(2.0_C_DOUBLE, str1%dfield, "pass_struct_by_value dfield")

    rvi = pass_struct_by_reference_in(str1) ! assign global_Cstruct1
    call assert_equals(6, rvi, "passStructByReferenceIn")
    call pass_struct_by_reference_out(str2) ! fetch global_Cstruct1
    call assert_equals(str1%ifield, str2%ifield, "passStructByReferenceOut")
    call assert_equals(str1%dfield, str2%dfield, "passStructByReferenceOut")

    ! Change str1 in place.
    call pass_struct_by_reference_inout(str1)
    call assert_equals(4, str1%ifield, "passStructByReferenceInOut")

  end subroutine test_struct

  subroutine test_default_args
    use cxxlibrary_mod
    real(C_DOUBLE) :: some_var(2)
    integer(C_INT) :: out1, out2
    
    call assert_true(default_ptr_is_null())
    call assert_false(default_ptr_is_null(some_var))

    ! flag defaults to false
    call default_args_in_out(1, out1, out2)
    call assert_equals(1, out1, "defaultArgsInOut")
    call assert_equals(2, out2, "defaultArgsInOut")
    call default_args_in_out(1, out1, out2, .true.)
    call assert_equals(1, out1, "defaultArgsInOut")
    call assert_equals(20, out2, "defaultArgsInOut")
    
  end subroutine test_default_args

  subroutine test_generic
    use cxxlibrary_mod
    character(30) rv

    rv = get_group_name(1)
    call assert_equals("global-string", rv, "getGroupName");
    rv = get_group_name(1_C_INT)
    call assert_equals("global-string", rv, "getGroupName");
    rv = get_group_name(1_C_LONG)
    call assert_equals("global-string", rv, "getGroupName");
    rv = get_group_name(1_C_INT32_T)
    call assert_equals("global-string", rv, "getGroupName");
    rv = get_group_name(1_C_INT64_T)
    call assert_equals("global-string", rv, "getGroupName");
    
  end subroutine test_generic
  
end program tester
