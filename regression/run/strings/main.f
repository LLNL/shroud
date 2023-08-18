! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
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
  call test_charargs
  call test_charargs_c
  call test_functions
  call test_string_array
  call test_explicit
  call char_functions

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_charargs
    ! test C++ functions

    character(30) str
    character ch

    call set_case_name("test_charargs")

    call pass_char("w")
    ch = return_char()
    call assert_equals("w", ch, "passChar/returnChar")

    call pass_char_force("x")
    ch = return_char()
    call assert_equals("x", ch, "passCharForce/returnChar")

    ! character(*) function
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    call pass_char_ptr(dest=str, src="bird")
    call assert_true( str == "bird")

    ! call C version directly via the interface
    ! caller is responsible for nulls
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    call c_pass_char_ptr(dest=str, src="mouse" // C_NULL_CHAR)
    call assert_true( str(1:5) == "mouse")
    call assert_true( str(6:6) == C_NULL_CHAR)

    str = 'dog'
    call pass_char_ptr_in_out(str)
    call assert_true( str == "DOG")

  end subroutine test_charargs

  subroutine test_charargs_c
    ! test extern "C" functions

    character(30) str
    character ch

    call set_case_name("test_charargs_c")

    call cpass_char("w")

    ch = creturn_char()
    call assert_equals("w", ch, "CreturnChar")

    ! character(*) function
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    call cpass_char_ptr(dest=str, src="bird")
    call assert_true( str == "bird")

    ! Test passing a blank string, treat as NULL pointer.
    ! +blanknull
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    call cpass_char_ptr(dest=str, src=" ")
    call assert_true( str == "NULL", "blank string")

    ! Test passing a blank string, treat as NULL pointer.
    ! options.F_blanknull
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    call cpass_char_ptr_blank(dest=str, src=" ")
    call assert_true( str == "NULL", "blank string")

    ! call C version directly via the interface
    ! caller is responsible for nulls
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    call c_cpass_char_ptr(dest=str, src="mouse" // C_NULL_CHAR)
    call assert_true( str(1:5) == "mouse")
    call assert_true( str(6:6) == C_NULL_CHAR)

  end subroutine test_charargs_c

  subroutine test_functions

    type(C_PTR) :: strptr
    character(len=:), allocatable :: astr
    character(len=:), pointer :: pstr
    character(30) str
    character(30), parameter :: static_str = "dog                         "
    character, pointer :: raw_str(:)
    integer(C_INT) :: nlen

    call set_case_name("test_functions")

    ! problem with pgi
    ! character(*) function
    astr = get_char_ptr1()
    call assert_true( astr == "bird", "get_char_ptr1")
    deallocate(astr)

    ! character(30) function
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    str = get_char_ptr2()
    call assert_true( str == "bird", "get_char_ptr2")

    ! string_result_as_arg
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    call get_char_ptr3(str)
    call assert_true( str == "bird", "get_char_ptr3")

    strptr = get_char_ptr4()
    call c_f_pointer(strptr, raw_str, [4])
    call assert_true( &
         raw_str(1) == "b" .and. &
         raw_str(2) == "i" .and. &
         raw_str(3) == "r" .and. &
         raw_str(4) == "d", "get_char_ptr4")

#ifdef HAVE_CHARACTER_POINTER_FUNCTION
    nullify(pstr)
    pstr => get_char_ptr5()
    call assert_true(associated(pstr), "get_char_ptr5 associated")
    call assert_true(len(pstr) == 4, "get_char_ptr5 len")
    call assert_true(pstr == "bird", "get_char_ptr5")
#endif

    !--------------------------------------------------

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

    ! call C version directly via the interface
    ! caller is responsible for nulls
    ! str must be long enough for the result from the function
    str = "cat" // C_NULL_CHAR
    call c_accept_string_reference(str)
    call assert_true( str(1:6) == "catdog")
    call assert_true( str(7:7) == C_NULL_CHAR)

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

    ! Call C++ function directly by adding trailing NULL.
    nlen = c_accept_string_instance("from Fortran" // C_NULL_CHAR)
    call assert_equals(12, nlen, "acceptStringInstance")

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

    ! copy into argument
    call fetch_array_string_arg(strs)
    call assert_true(strs(1) == "apple",  "fetch_array_string_copy(1)")
    call assert_true(strs(2) == "pear",   "fetch_array_string_copy(2)")
    call assert_true(strs(3) == "peach",  "fetch_array_string_copy(3)")
    call assert_true(strs(4) == "cherry", "fetch_array_string_copy(4)")
    call assert_true(strs(5) == " ",      "fetch_array_string_copy(5)")

    ! allocate the argument
    call assert_false(allocated(stralloc), "stralloc not allocated")
    call fetch_array_string_alloc(stralloc)
    call assert_true(allocated(stralloc), "stralloc is allocated")
    call assert_equals(4, size(stralloc), "size of stralloc")
    call assert_equals(6, len(stralloc), "len of stralloc")
    call assert_true(stralloc(1) == "apple",  "fetch_array_string_alloc(1)")
    call assert_true(stralloc(2) == "pear",   "fetch_array_string_alloc(2)")
    call assert_true(stralloc(3) == "peach",  "fetch_array_string_alloc(3)")
    call assert_true(stralloc(4) == "cherry", "fetch_array_string_alloc(4)")
    
  end subroutine test_string_array
  
  subroutine test_explicit
    character(10) name
    call set_case_name("test_explicit")

    name = "cat"
    call explicit1(name)

    name = " "
    call explicit2(name)
    call assert_equals("a", name(1:1))
    
  end subroutine test_explicit

  subroutine char_functions
    character(20), target :: str
    character(20) :: str1, str2
    
    call set_case_name("test_explicit")

    call assert_equals(4, cpass_char_ptr_notrim("tree"), "CpassCharPtrNotrim")

    ! CpassCharPtrCAPI should get two equal pointers.
    str = " "
    call assert_equals(1, cpass_char_ptr_capi(c_loc(str), str), "CpassCharPtrCAPI")

    str1 = "sample string"
    str2 = str1
    call assert_equals(1, cpass_char_ptr_capi2(str1, str2))
    
  end subroutine char_functions

end program tester
