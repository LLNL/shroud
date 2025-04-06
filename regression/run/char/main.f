! Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from char.yaml.
!
#include "shroud/features.h"

program tester
  use fruit
  use iso_c_binding
  use char_mod
  implicit none
  logical ok

  call init_fruit

  call init_test
  call test_charargs
  call test_charargs_c
  call test_functions
  call test_functions_as_arg
  call test_explicit
  call char_functions
#ifdef TEST_C_WRAPPER
  call test_c_wrapper
#endif
  call test_char_arrays
  call test_char_ptr_out

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
    character(4)  str4
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
    call assert_true(str == "bird", "passCharPtr")

    ! character(*) function
    ! Input and output the same length.
    str4 = 'XXXX'
! XXX - asan error
!    call pass_char_ptr(dest=str4, src="bird")
!    call assert_true(str4 == "bird", "passCharPtr - same length")

    str = 'dog'
    call pass_char_ptr_in_out(str)
    call assert_true(str == "DOG", "passCharPtrInOut")

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
    call assert_true( str == "bird", "CpassCharPtr")

    ! Test passing a blank string, treat as NULL pointer.
    ! +blanknull
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    call cpass_char_ptr(dest=str, src=" ")
    call assert_true( str == "NULL", "CpassCharPtr - blank")

    ! Test passing a blank string, treat as NULL pointer.
    ! options.F_blanknull
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    call cpass_char_ptr_blank(dest=str, src=" ")
    call assert_true( str == "NULL", "CpassCharPtrBlank - blank")

  end subroutine test_charargs_c

  subroutine test_functions

    type(C_PTR) :: strptr
    character(len=:), allocatable :: astr
    character(len=:), pointer :: pstr
    character(30) str
    character(30), parameter :: static_str = "dog                         "
    character, pointer :: raw_str(:)

    call set_case_name("test_functions")

    ! problem with pgi
    ! character(*) function
    astr = get_char_ptr1()
    call assert_true( astr == "bird", "get_char_ptr1")
    deallocate(astr)

    ! character(30) function
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    str = get_const_char_ptr_len()
    call assert_true( str == "bird", "getConstCharPtrLen")

!    ! string_result_as_arg
!    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
!    call get_const_char_ptr_as_arg(str)
!    call assert_true( str == "bird", "getConstCharPtrAsArg")

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

  end subroutine test_functions

! Return function result as an argument
  subroutine test_functions_as_arg

!    type(C_PTR) :: strptr
    character(len=:), allocatable :: astr
!    character(len=:), pointer :: pstr
    character(30) str
    character(30), parameter :: static_str = "dog                         "
!    character, pointer :: raw_str(:)

    call set_case_name("test_functions_as_arg")

    call  get_const_char_ptr_as_alloc_arg(astr)
    call assert_true(astr == "bird", "GetConstCharPtrAsAllocArg")
    deallocate(astr)

    ! character(30) function
!    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
!    str = get_const_char_ptr_len()
!    call assert_true( str == "bird", "getConstCharPtrLen")

    ! string_result_as_arg
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    call get_const_char_ptr_as_arg(str)
    call assert_true( str == "bird", "getConstCharPtrAsArg")

!    strptr = get_char_ptr4()
!    call c_f_pointer(strptr, raw_str, [4])
!    call assert_true( &
!         raw_str(1) == "b" .and. &
!         raw_str(2) == "i" .and. &
!         raw_str(3) == "r" .and. &
!         raw_str(4) == "d", "get_char_ptr4")

!#ifdef HAVE_CHARACTER_POINTER_FUNCTION
!    nullify(pstr)
!    pstr => get_char_ptr5()
!    call assert_true(associated(pstr), "get_char_ptr5 associated")
!    call assert_true(len(pstr) == 4, "get_char_ptr5 len")
!    call assert_true(pstr == "bird", "get_char_ptr5")
!#endif

  end subroutine test_functions_as_arg

  subroutine test_explicit
    character(10) name
    call set_case_name("test_explicit")

    name = "cat"
    call explicit1(name)

    name = " "
    call explicit2(name)
    call assert_equals("a", name(1:1), "explicit2")
    
  end subroutine test_explicit

  subroutine char_functions
    character(20), target :: str
    character(20) :: str1, str2
    
    call set_case_name("char_functions")

    call assert_equals(4, cpass_char_ptr_notrim("tree"), "CpassCharPtrNotrim")

    ! CpassCharPtrCAPI should get two equal pointers.
    str = " "
    call assert_equals(1, cpass_char_ptr_capi(c_loc(str), str), "CpassCharPtrCAPI")

    str1 = "sample string"
    str2 = str1
    call assert_equals(1, cpass_char_ptr_capi2(str1, str2), "CpassCharPtrCAPI2")
    
  end subroutine char_functions

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

    ! call C version directly via the interface
    ! caller is responsible for nulls
    str = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    call c_cpass_char_ptr(dest=str, src="mouse" // C_NULL_CHAR)
    call assert_true( str(1:5) == "mouse", "CpassCharPtr 1")
    call assert_true( str(6:6) == C_NULL_CHAR, "CpassCharPtr 1")

  end subroutine test_c_wrapper
#endif
  
  subroutine test_char_arrays
    integer nchar, i
    character(10) :: in(3) = [ &
         "dog       ", &
         "cat       ", &
         "monkey    "  &
         ]
    character(11), target :: inc(3)
    type(C_PTR) :: inptr(3)

    call set_case_name("test_char_arrays")

    ! Call the bufferify function.
    ! It will copy strings to create char ** variable.
    nchar = accept_char_array_in(in)
    call assert_equals(len_trim(in(1)), nchar, "acceptCharArrayIn")

    ! Create the char** data structure.
    do i = 1, 3
       inc(i) = trim(in(i)) // C_NULL_CHAR
       inptr(i) = C_LOC(inc(i))
    enddo

    ! The interface is not created with pointers-c
    nchar = c_accept_char_array_in(inptr)
    call assert_equals(len_trim(in(1)), nchar, "acceptCharArrayIn")

  end subroutine test_char_arrays

  subroutine test_char_ptr_out
    integer irv;
    character(len=20) :: outstr
    character(len=6) :: outstrshort
    character(len=:), pointer :: outptr
    character(8), target :: nonnull = "Non-null"

    call set_case_name("test_char_ptr_out")

    call fetch_char_ptr_copy_library(outstr)
    call assert_equals("static_char_array", outstr, "fetchCharPtrCopyLibrary")

    ! Make sure it does not overrun the argument.
    call fetch_char_ptr_copy_library(outstrshort)
    call assert_equals("static", outstrshort, "fetchCharPtrCopyLibrary short")

    nullify(outptr)
    call fetch_char_ptr_library(outptr)
    call assert_true(associated(outptr), "fetchCharPtrLibrary associated")
    call assert_equals("static_char_array", outptr, "fetchCharPtrLibrary value")
    
    outptr => nonnull
    irv = fetch_char_ptr_library_null(outptr)
    call assert_false(associated(outptr), "fetchCharPtrLibraryNULL")
    call assert_equals(0, irv, "fetchCharPtrLibrary result")
    
  end subroutine test_char_ptr_out
  
end program tester
