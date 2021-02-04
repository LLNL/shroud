! Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from clibrary.yaml.
!
subroutine incr2_int(input)
  use iso_c_binding
  implicit none
  integer(C_INT) :: input
  input = input + 20
end subroutine incr2_int

subroutine incr2_double(input)
  use iso_c_binding
  implicit none
  real(C_DOUBLE) :: input
  input = input + 20.5_C_DOUBLE
end subroutine incr2_double

subroutine incr3_int(input)
  use iso_c_binding
  implicit none
  integer(C_INT) :: input
  input = input + 20
end subroutine incr3_int

subroutine incr3_double(input)
  use iso_c_binding
  implicit none
  real(C_DOUBLE) :: input
  input = input + 20.5_C_DOUBLE
end subroutine incr3_double

module callback_mod
contains
  subroutine incr2b_int(input)
    use iso_c_binding
    implicit none
    integer(C_INT) :: input
    input = input + 20
  end subroutine incr2b_int

  subroutine incr2b_double(input)
    use iso_c_binding
    implicit none
    real(C_DOUBLE) :: input
    input = input + 20.5_C_DOUBLE
  end subroutine incr2b_double
  
  subroutine incr3b_int(input)
    use iso_c_binding
    implicit none
    integer(C_INT) :: input
    input = input + 20
  end subroutine incr3b_int

  subroutine incr3b_double(input)
    use iso_c_binding
    implicit none
    real(C_DOUBLE) :: input
    input = input + 20.5_C_DOUBLE
  end subroutine incr3b_double

  ! On Intel, bind(C) is required because of the VALUE attribute.
  subroutine set_alloc(tc, arr) bind(C)
    use iso_c_binding, only : C_INT
    use clibrary_mod, only : array_info
    integer(C_INT), intent(IN), value :: tc
    type(array_info), intent(INOUT) :: arr
    arr%tc = tc
  end subroutine set_alloc
  
end module callback_mod

program tester
  use fruit
  use iso_c_binding
  use clibrary_mod
  implicit none
  real(C_DOUBLE), parameter :: pi = 3.1415926_C_DOUBLE
  integer, parameter :: lenoutbuf = 40
  logical ok

  logical rv_logical, wrk_logical
!  integer rv_integer
  integer(C_INT) rv_int
  real(C_DOUBLE) rv_double
  character(30) rv_char

  call init_fruit

  call test_functions
  call test_callback

!  call test_vector

!  call test_class1

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_functions
    integer(C_INT), target :: int_var
    character(MAXNAME) name1, name2
    character(lenoutbuf)  :: outbuf
    character(30) str
    type(C_PTR) :: cptr1, cptr2

    integer(C_INT) int_array(10)
    real(C_DOUBLE) double_array(2,5)

    call set_case_name("test_functions")

    call no_return_no_arguments
    call assert_true(.true.)

    rv_double = pass_by_value(1.d0, 4)
    call assert_true(rv_double == 5.d0)

    ! The C macro force the first argument to be 1.0
    rv_double = pass_by_value_macro(4)
    call assert_true(rv_double == 5.d0)

    call pass_by_reference(3.14d0, int_var)
    call assert_equals(3, int_var)

    rv_logical = .true.
    wrk_logical = .true.
    call check_bool(.true., rv_logical, wrk_logical)
    call assert_false(rv_logical)
    call assert_false(wrk_logical)

    rv_logical = .false.
    wrk_logical = .false.
    call check_bool(.false., rv_logical, wrk_logical)
    call assert_true(rv_logical)
    call assert_true(wrk_logical)

    call assert_true(function4a("dog", "cat") == "dogcat")

    call accept_name("spot")
!    call assert_true(last_function_called() == "acceptName")

    str = 'dog'
    call pass_char_ptr_in_out(str)
    call assert_true( str == "DOG")
    !--------------------------------------------------

    name1 = " "
    call return_one_name(name1)
    call assert_equals("bill", name1)

    name1 = " "
    name2 = " "
    call return_two_names(name1, name2)
    call assert_equals("tom", name1)
    call assert_equals("frank", name2)

    !--------------------------------------------------

    call implied_text_len(name1)
    call assert_equals("ImpliedTextLen", name1)

    rv_int = implied_len("bird")
    call assert_true(rv_int == 4)
    rv_int = implied_len_trim("bird")
    call assert_true(rv_int == 4)

    rv_char = "bird"
    rv_int = implied_len(rv_char)
    call assert_true(rv_int == len(rv_char))
    rv_int = implied_len_trim(rv_char)
    call assert_true(rv_int == len_trim(rv_char))

    call assert_true(implied_bool_true())
    call assert_false(implied_bool_false())

    cptr1 = c_loc(int_var)
    cptr2 = C_NULL_PTR
    call pass_void_star_star(cptr1, cptr2)
    call assert_true(c_associated(cptr1, cptr2))

    rv_int = pass_assumed_type(23_C_INT)
    call assert_equals(23, rv_int)
    rv_int = pass_assumed_type_buf(33_C_INT, outbuf)
    call assert_equals(33, rv_int)

    call pass_assumed_type_dim(int_array)
    call pass_assumed_type_dim(double_array)

!    call function4b("dog", "cat", rv_char)
!    call assert_true( rv_char == "dogcat")
!
!    call assert_equals(function5(), 13.1415d0)
!    call assert_equals(function5(1.d0), 11.d0)
!    call assert_equals(function5(1.d0, .false.), 1.d0)
!
!    call function6("name")
!    call assert_true(last_function_called() == "Function6(string)")
!    call function6(1)
!    call assert_true(last_function_called() == "Function6(int)")
!
!    call function9(1.0)
!    call assert_true(.true.)
!    call function9(1.d0)
!    call assert_true(.true.)
!
!    call function10()
!    call assert_true(.true.)
!    call function10("foo", 1.0e0)
!    call assert_true(.true.)
!    call function10("bar", 2.0d0)
!    call assert_true(.true.)

!    rv_int = typefunc(2)
!    call assert_true(rv_int .eq. 2)
!
!    rv_int = enumfunc(1)
!    call assert_true(rv_int .eq. 2)

  end subroutine test_functions

  subroutine test_callback
    use callback_mod
    integer(C_INT) arg_int
    real(C_DOUBLE) arg_dbl
    character(lenoutbuf)  :: outbuf
    type(array_info) :: arr
    external incr2_int, incr2_double
    external incr3_int, incr3_double

    call set_case_name("test_callback")

    ! incr_int matches the prototype in the YAML file.
    ! incr_double, does not.

    !----------
    ! callback2, accept any type of function.
    ! first argument, tells C how to cast pointer.
    arg_int = 10_C_INT
    call callback2(1, arg_int, incr2_int)
    call assert_equals(30, arg_int, "incr2_int")

    arg_dbl = 3.4_C_DOUBLE
    call callback2(2, arg_dbl, incr2_double)
    call assert_equals(23.9_C_DOUBLE, arg_dbl, "incr2_int")

    !----------
    ! callback3, accept any type of function.
    ! first argument, tells C how to cast pointer.
    arg_int = 10_C_INT
    call callback3("int", arg_int, incr3_int, outbuf)
    call assert_equals(30, arg_int, "incr3_int")

    arg_dbl = 3.4_C_DOUBLE
    call callback3("double", arg_dbl, incr3_double, outbuf)
    call assert_equals(23.9_C_DOUBLE, arg_dbl, "incr3_double")

    !----------
    ! routines from a module, with an implicit interface.
    ! callback2, accept any type of function.
    ! first argument, tells C how to cast pointer.
    arg_int = 10_C_INT
    call callback2(1, arg_int, incr2b_int)
    call assert_equals(30, arg_int, "incr2b_int")

    arg_dbl = 3.4_C_DOUBLE
    call callback2(2, arg_dbl, incr2b_double)
    call assert_equals(23.9_C_DOUBLE, arg_dbl, "incr2b_double")

    !----------
    ! callback3, accept any type of function.
    ! first argument, tells C how to cast pointer.
    arg_int = 10_C_INT
    call callback3("int", arg_int, incr3b_int, outbuf)
    call assert_equals(30, arg_int, "incr3b_int")

    arg_dbl = 3.4_C_DOUBLE
    call callback3("double", arg_dbl, incr3b_double, outbuf)
    call assert_equals(23.9_C_DOUBLE, arg_dbl, "incr3b_double")

    ! The callback sets tc
    arr%tc = 0
    call callback_set_alloc(3, arr, set_alloc)
    call assert_equals(3, arr%tc, "callback_set_alloc")
    

  end subroutine test_callback

!  subroutine test_vector
!    integer(C_INT) intv(5)
!    character(10) :: names(3)
!    integer irv
!
!    call set_case_name("test_vector")
!
!    intv = [1,2,3,4,5]
!    irv = vector_sum(intv)
!    call assert_true(irv .eq. 15)
!
!    intv(:) = 0
!    call vector_iota(intv)
!    call assert_true(all(intv(:) .eq. [1,2,3,4,5]))
!
!    intv = [1,2,3,4,5]
!    call vector_increment(intv)
!    call assert_true(all(intv(:) .eq. [2,3,4,5,6]))
!
!    ! count number of underscores
!    names = [ "dog_cat   ", "bird_mouse", "__        " ]
!    irv = vector_string_count(names)
!    call assert_true(irv == 4)
!
!    ! Fill strings into names
!    names = " "
!    irv = vector_string_fill(names)
!    call assert_true(irv == 2)
!    call assert_true( names(1) == "dog")
!    call assert_true( names(2) == "bird")
!    call assert_true( names(3) == " ")
!
!    ! Append -like to names.
!    ! Note that strings will be truncated to len(names)
!    names = [ "fish      ", "toolong   ", "          " ]
!    call vector_string_append(names)
!    call assert_true( names(1) == "fish-like")
!    call assert_true( names(2) == "toolong-li")
!    call assert_true( names(3) == "-like")
! 
!  end subroutine test_vector

!  subroutine test_class1
!    type(class1) obj
!
!    call set_case_name("test_class1")
!
!    obj = class1_new()
!    call assert_true(c_associated(obj%get_instance()), "class1_new")
!
!    call obj%method1
!    call assert_true(.true.)
!
!    call useclass(obj)
!
!    call obj%delete
!    call assert_true(.not. c_associated(obj%get_instance()), "class1_delete")
!  end subroutine test_class1

end program tester
