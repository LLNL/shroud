! Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from strings.yaml.
!
#include "shroud/features.h"

program tester
  use fruit
  use iso_c_binding
  use strings_mod
  implicit none
  logical ok

  call init_fruit

  call init_test
  call test_charargs_c
  call test_functions
  call test_string_array
#ifdef TEST_C_WRAPPER
  call test_c_wrapper
#endif

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_charargs_c
    ! test extern "C" functions

    character ch

    call set_case_name("test_charargs_c")

    call cpass_char("w")

    ch = creturn_char()
    call assert_equals("w", ch, "CreturnChar")

  end subroutine test_charargs_c

  subroutine test_functions

    character(len=:), pointer :: pstr
    character(30) str
    character(30), parameter :: static_str = "dog                         "
    integer(C_INT) :: nlen

    call set_case_name("test_functions")

    ! character(:), allocatable function
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    str = get_const_string_result()
    call assert_true(str == "getConstStringResult", "getConstStringResult")

    ! character(30) function
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    str = get_const_string_len()
    call assert_true(str == static_str, "getConstStringLen")

    ! string_result_as_arg
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    call get_const_string_as_arg(str)
    call assert_true(str == static_str, "getConstStringAsArg")

    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    str = get_const_string_alloc()
    call assert_true(str == "getConstStringAlloc", "getConstStringAlloc")
 
    !--------------------------------------------------

    ! problem with pgi
    ! character(*) function
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    str = get_const_string_ref_pure()
    call assert_true( str == static_str, "getConstStringRefPure")

    ! character(30) function
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    str = get_const_string_ref_len()
    call assert_true( str == static_str, "getConstStringRefLen")

    ! string_result_as_arg
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    call get_const_string_ref_as_arg(str)
    call assert_true( str == static_str, "getConstStringRefAsArg")
 
    ! character(30) function
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    str = get_const_string_ref_len_empty()
    call assert_true( str == " ", "getConstStringRefLenEmpty")

    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    str = get_const_string_ref_alloc()
    call assert_true( str == static_str, "getConstStringRefAlloc")

    !--------------------------------------------------

    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    str = get_const_string_ptr_len()
    call assert_true(str == "getConstStringPtrLen", "getConstStringPtrLen")

    ! string_result_as_arg
 
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    str = get_const_string_ptr_alloc()
    call assert_true( str == static_str, "getConstStringPtrAlloc")

    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    str = get_const_string_ptr_owns_alloc()
    call assert_true( str == "getConstStringPtrOwnsAlloc", &
         "getConstStringPtrOwnsAlloc")

    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    str = get_const_string_ptr_owns_alloc_pattern()
    call assert_true( str == "getConstStringPtrOwnsAllocPatt", &
         "getConstStringPtrOwnsAllocPattern")

    !--------------------------------------------------
    ! POINTER result

#ifdef HAVE_CHARACTER_POINTER_FUNCTION
    nullify(pstr)
    pstr => get_const_string_ptr_pointer()
    call assert_true(associated(pstr), "getConstStringPtrPointer associate")
    call assert_true(pstr == static_str, "getConstStringPtrPointer")
#endif

!    pstr => get_const_string_ptr_owns_pointer()
!    call assert_true( str == "getConstStringPtrOwnsPointer", &
!         "getConstStringPtrOwnsPointer")
    
    !--------------------------------------------------

    call accept_string_const_reference("cat")
!    check global_str == "cat"

    str = " "
    call accept_string_reference_out(str)
    call assert_true( str == "dog")

    str = "cat"
    call accept_string_reference(str)
    call assert_true( str == "catdog")

    ! Store in global_str.
    call accept_string_pointer_const("from Fortran")

    ! Fetch from global_str.
    call fetch_string_pointer(str)
    call assert_true( str == "from Fortran", "fetchStringPointer")

    call fetch_string_pointer_len(str, nlen)
    call assert_true( str == "from Fortran", "FetchStringPointerLen")
    call assert_equals(len_trim(str), nlen, "FetchStringPointerLen")

    ! Return length of string
    nlen = accept_string_instance("from Fortran")
    call assert_equals(12, nlen, "acceptStringInstance")
    str = "from Fortran"
    nlen = accept_string_instance(str) ! Returns trimmed length
    call assert_equals(12, nlen, "acceptStringInstance")
    ! argument is passed by value to C++ so changes will not effect argument.
    call assert_equals("from Fortran", str)

    ! append "dog".
    str = "bird"
    call accept_string_pointer(str)
    call assert_true( str == "birddog", "acceptStringPointer")

    str = "bird"
    call accept_string_pointer_len(str, nlen)
    call assert_true( str == "birddog", "acceptStringPointerLen")
    call assert_equals(len_trim(str), nlen, "acceptStringPointerLen")

  end subroutine test_functions

  subroutine test_string_array
    character(20) :: strs(5)
    character(:), allocatable :: stralloc(:)

    call set_case_name("test_string_array")

    ! Copy into argument.
    strs = "xxx"
    call fetch_array_string_arg(strs)
    call assert_equals("apple",  strs(1), "fetchArrayStringArg(1)")
    call assert_equals("pear",   strs(2), "fetchArrayStringArg(2)")
    call assert_equals("peach",  strs(3), "fetchArrayStringArg(3)")
    call assert_equals("cherry", strs(4), "fetchArrayStringArg(4)")
    call assert_equals(" ",      strs(5), "fetchArrayStringArg(5)")

    ! Allocate the argument.
    call assert_false(allocated(stralloc), "stralloc not allocated")
    call fetch_array_string_alloc(stralloc)
    call assert_true(allocated(stralloc), "stralloc is allocated")
    call assert_equals(4, size(stralloc), "size of stralloc")
    call assert_equals(6, len(stralloc), "len of stralloc")
!    print *, "XXXX", len(stralloc), size(stralloc)
!    print *, "XXXX", stralloc
    call assert_equals("apple",  stralloc(1), "fetchArrayStringAlloc(1)")
    call assert_equals("pear",   stralloc(2), "fetchArrayStringAlloc(2)")
    call assert_equals("peach",  stralloc(3), "fetchArrayStringAlloc(3)")
    call assert_equals("cherry", stralloc(4), "fetchArrayStringAlloc(4)")
    deallocate(stralloc)
    
    ! Allocate the argument with a predefined len.
    ! The allocate uses a fixed size of 20.
    call assert_false(allocated(stralloc), "stralloc not allocated")
    call fetch_array_string_alloc_len(stralloc)
    call assert_true(allocated(stralloc), "stralloc is allocated")
    call assert_equals(4, size(stralloc), "size of stralloc")
    call assert_equals(20, len(stralloc), "len of stralloc")
    call assert_equals("apple",  stralloc(1), "fetchArrayStringAllocLen(1)")
    call assert_equals("pear",   stralloc(2), "fetchArrayStringAllocLen(2)")
    call assert_equals("peach",  stralloc(3), "fetchArrayStringAllocLen(3)")
    call assert_equals("cherry", stralloc(4), "fetchArrayStringAllocLen(4)")
    
  end subroutine test_string_array
  
#ifdef TEST_C_WRAPPER
  ! Calling C only wrappers from Fortran via an interface
  subroutine test_c_wrapper
    character(30) str
    integer(C_INT) :: nlen

    call set_case_name("test_c_wrapper")

    ! call C version directly via the interface
    ! caller is responsible for nulls
    ! str must be long enough for the result from the function
    str = "cat" // C_NULL_CHAR
    call c_accept_string_reference(str)
    call assert_true( str(1:6) == "catdog", "acceptStringReference 1")
    call assert_true( str(7:7) == C_NULL_CHAR, "acceptStringReference 2")

    ! Call C++ function directly by adding trailing NULL.
    nlen = c_accept_string_instance("from Fortran" // C_NULL_CHAR)
    call assert_equals(12, nlen, "acceptStringInstance")

  end subroutine test_c_wrapper
#endif
  
end program tester
