! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from cxxlibrary.yaml.
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
  call test_nested
  call test_return_this

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

    call set_case_name("struct")

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
    
    call set_case_name("default_args")

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

    call set_case_name("generic")

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

  subroutine test_nested
    use cxxlibrary_mod

    type(nested) pnode
    type(nested) :: n1
    type(nested), target :: kids(3)
    type(nested), pointer :: parent, single, array(:)
    type(C_PTR) :: kidsaddr(3)
    type(C_PTR), pointer :: child(:)

    call set_case_name("nested")

    pnode%index = 1
    n1%index = 2
    kids(1)%index = 31
    kids(2)%index = 32
    kids(3)%index = 33

    call nested_set_parent(n1, pnode)
    
    parent => nested_get_parent(n1)
    call assert_equals(pnode%index, parent%index, "nested_get_parent 1");

    n1%sublevels = size(kids)

    ! Setting  nested **child field
    kidsaddr(1) = c_loc(kids(1))
    kidsaddr(2) = c_loc(kids(2))
    kidsaddr(3) = c_loc(kids(3))
    call nested_set_child(n1, kidsaddr)
    child => nested_get_child(n1)
    call assert_true(associated(child), "nested_get_child associated")
    call assert_equals(n1%sublevels, size(child), "nested_get_child size")
    call c_f_pointer(child(1), single)
    call assert_equals(single%index, kids(1)%index, "nested_get_child 1");
    call c_f_pointer(child(2), single)
    call assert_equals(single%index, kids(2)%index, "nested_get_child 2");
    call c_f_pointer(child(3), single)
    call assert_equals(single%index, kids(3)%index, "nested_get_child 3");

    ! Setting nested *array field
    call nested_set_array(n1, kids)
    array => nested_get_array(n1)
    call assert_true(associated(array), "nested_get_array associated")
    call assert_equals(n1%sublevels, size(array), "nested_get_array size")
    call assert_equals(array(1)%index, kids(1)%index, "nested_get_array 1");
    call assert_equals(array(2)%index, kids(2)%index, "nested_get_array 2");
    call assert_equals(array(3)%index, kids(3)%index, "nested_get_array 3");
    
  end subroutine test_nested

  subroutine test_return_this
    use cxxlibrary_mod
    type(class1) obj
    integer(C_INT) length

    call set_case_name("return_this")

    obj = class1()

    length = obj%check_length()
    call assert_equals(1, length, "check_length no args");
    length = obj%check_length(12_C_INT)
    call assert_equals(12, length, "check_length int");
    length = obj%check_length(13_C_LONG)
    call assert_equals(13, length, "check_length long");
    
    length = obj%get_length()
    call assert_equals(99, length, "ctor length");
    
    call obj%declare(5)
    length = obj%get_length()
    call assert_equals(1, length, "default length");

    call obj%declare(5, 33_C_INT)
    length = obj%get_length()
    call assert_equals(33, length, "explicit length in");

    call obj%declare(5, 44_C_LONG)
    length = obj%get_length()
    call assert_equals(44, length, "explicit length long");
    
  end subroutine test_return_this

  
end program tester
