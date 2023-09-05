! Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from tutorial.yaml.
!
function incr2(input) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int), value :: input
  integer(c_int) :: incr2
  incr2 = input + 20
end function incr2

program tester
  use fruit
  use iso_c_binding
  use tutorial_mod
  implicit none
  logical ok

  integer rv_integer
  integer(C_INT) rv_int
  real(C_DOUBLE) rv_double

  call init_fruit

  call test_enums
  call test_functions

  call test_callback

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_enums
    ! test values of enumerations
    integer(C_INT) rv_int

    call set_case_name("test_enums")

    call assert_equals(0, red, "tutorial_color_red")
    call assert_equals(1, blue, "tutorial_color_blue")
    call assert_equals(2, white, "tutorial_color_white")

    rv_int = colorfunc(BLUE)
    call assert_true(rv_int .eq. RED, "tutorial_color_RED")

  end subroutine test_enums

  subroutine test_functions

    integer(C_INT) :: minout, maxout
!    character(len=:), allocatable :: rv4c

    call set_case_name("test_functions")

    call no_return_no_arguments
    call assert_true(.true., "no_return_no_arguments")

    rv_double = pass_by_value(1.d0, 4)
    call assert_true(rv_double == 5.d0, "pass_by_value")

!    call assert_true( function4a("dog", "cat") == "dogcat", "function4a")

!    call function4b("dog", "cat", rv_char)
!    call assert_true( rv_char == "dogcat", "function4b")

    call assert_equals( "dawgkat", concatenate_strings("dawg", "kat"), "concatenate_strings")

! warning: '.rv4c' may be used uninitialized in this function [-Wmaybe-uninitialized]
! gfortran 4.9.3
!    call assert_false(allocated(rv4c))
!    rv4c = concatenate_strings("one", "two")
!    call assert_true(allocated(rv4c))
!    call assert_true(len(rv4c) == 6)
!    call assert_true(rv4c == "onetwo")
!    deallocate(rv4c)

!    call assert_true( function4d() == "Function4d", "function4d")

    call assert_equals(13.1415d0, use_default_arguments(), &
         "UseDefaultArguments 1")
    call assert_equals(11.d0, use_default_arguments(1.d0), &
         "UseDefaultArguments 2")
    call assert_equals(1.d0, use_default_arguments(1.d0, .false.), &
         "UseDefaultArguments 3")

    call overloaded_function("name")
    call assert_true(last_function_called() == "OverloadedFunction(string)", &
         "OverloadedFunction 1")
    call overloaded_function(1)
    call assert_true(last_function_called() == "OverloadedFunction(int)", &
         "OverloadedFunction 2")

    call template_argument(1)
    call assert_true(last_function_called() == "TemplateArgument<int>",  &
         "TemplateArgument<int>")
    call template_argument(10.d0)
    call assert_true(last_function_called() == "TemplateArgument<double>", &
         "TemplateArgument<double>")

    ! return values set by calls to function7
    rv_integer = template_return_int()
    call assert_true(rv_integer == 1, "FunctionReturn<int>")
    rv_double = template_return_double()
    call assert_true(rv_double == 10.d0, "FunctionReturn<double>")

    call fortran_generic_overloaded()
    call assert_true(.true., "FortranGenericOverloaded 1")
    call fortran_generic_overloaded("foo", 1.0e0)
    call assert_true(.true., "FortranGenericOverloaded 2")
    call fortran_generic_overloaded("bar", 2.0d0)
    call assert_true(.true., "FortranGenericOverloaded 3")

    rv_int = use_default_overload(10)
    call assert_true(rv_int .eq. 10, "UseDefaultOverload 1")
    rv_int = use_default_overload(1.0d0, 10)
    call assert_true(rv_int .eq. 10, "UseDefaultOverload 2")

    rv_int = use_default_overload(10, 11, 12)
    call assert_true(rv_int .eq. 142, "UseDefaultOverload 3")
    rv_int = use_default_overload(1.0d0, 10, 11, 12)
    call assert_true(rv_int .eq. 142, "UseDefaultOverload 4")

    rv_int = typefunc(2)
    call assert_true(rv_int .eq. 2, "typefunc")

    rv_int = enumfunc(1)
    call assert_true(rv_int .eq. 2, "enumfunc")

    call get_min_max(minout, maxout)
    call assert_equals(-1, minout, "get_min_max minout")
    call assert_equals(100, maxout, "get_min_max maxout")

  end subroutine test_functions

  subroutine test_callback

    integer irv
    interface
       function incr2(input) bind(C)
         use iso_c_binding
         integer(c_int), value :: input
         integer(c_int) :: incr2
       end function incr2
    end interface

    call set_case_name("test_callback")

    irv = callback1(2, incr2)
    call assert_true(irv == 22)

  end subroutine test_callback

end program tester
