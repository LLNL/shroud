! Copyright Shroud Project Developers. See LICENSE file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from funptr.yaml.
!

module state
  ! shared by test and callbacks.
  use iso_c_binding
  implicit none

  integer, target :: counter

  ! callback3
  integer(C_INT) ival
  real(C_DOUBLE) dval
end module state

!----------------------------------------------------------------------
! external routines for function pointer arguments.
! These have no interfaces.

subroutine incr1_noiface()
  use state
  implicit none
  counter = counter + 1
end subroutine incr1_noiface

!----------

module callback_mod
  use iso_c_binding
  implicit none

  type(C_PTR) old_ptr
  integer(C_INT) old_int, old_int_array(10)
  character old_char
  character(80) old_string
  logical old_bool, old_bool_array
  
contains
  subroutine incr1() bind(C)
    ! A bind(C) is required for the intel/oneapi/ibm compiler since
    ! it is passed directly to C via a bind(C) interface.
    use state
    implicit none
    counter = counter + 1
  end subroutine incr1

  subroutine incr1_external()
    ! Note that bind(C) is not required with an EXTERNAL statement.
    ! But no interface checking can be done by the compiler.
    use state
    implicit none
    counter = counter + 1
  end subroutine incr1_external

  subroutine incr1_funptr()
    ! The user is required to pass C_FUNLOC(incr1_funptr)
    use state
    implicit none
    counter = counter + 1
  end subroutine incr1_funptr

!----------  

  subroutine incr2(i, j) bind(C)
    use iso_c_binding
    use state
    use funptr_mod, only : type_id
    implicit none
    integer(C_INT), value :: i
    integer(type_id), value :: j
    if (j == 1) counter = i
  end subroutine incr2

  subroutine incr2_d(i) bind(C)
    use iso_c_binding
    use state
    implicit none
    real(C_DOUBLE), value :: i
    counter = int(i)
  end subroutine incr2_d

  function incr2_fun(i) bind(C) result(rv)
    use iso_c_binding
    use state
    implicit none
    integer(C_INT) :: rv
    integer(C_INT), value :: i
    counter = i
    rv = counter
  end function incr2_fun

!----------

  subroutine incr3_int(in) bind(C)
    use iso_c_binding
    use state
    integer(C_INT), value :: in
    ival = in
    dval = 0.0
  end subroutine incr3_int

  subroutine incr3_double(in) bind(C)
    use iso_c_binding
    use state
    real(C_DOUBLE), value :: in
    ival = 0
    dval = in
  end subroutine incr3_double

!----------

  function sum4(ilow, nargs) bind(C)
    use iso_c_binding, only : C_INT
    implicit none
    integer(C_INT), intent(IN) :: ilow(*)
    integer(C_INT), value, intent(IN) :: nargs
    integer(C_INT) :: sum4
    integer i
    sum4 = ilow(1)
    do i = 2, nargs
       sum4 = sum4 + ilow(i)
    enddo
  end function sum4

  function product4(ilow, nargs) bind(C)
    use iso_c_binding, only : C_INT
    implicit none
    integer(C_INT), intent(IN) :: ilow(*)
    integer(C_INT), value, intent(IN) :: nargs
    integer(C_INT) :: product4
    integer i
    product4 = ilow(1)
    do i = 2, nargs
       product4 = product4 * ilow(i)
    enddo
  end function product4

!----------

  ! Return C_PTR to counter
  function set_ptr1() bind(C)
    use iso_c_binding
    use state
    type(C_PTR) :: set_ptr1
    set_ptr1 = C_LOC(counter)
  end function set_ptr1

  ! Return C_PTR to counter
  function set_double1(i1, i2) result(rvd) bind(C)
    use iso_c_binding
    use state
    integer(C_INT), value :: i1, i2
    real(C_DOUBLE) :: rvd
    rvd = i1 + i2
    dval = rvd
  end function set_double1
!----------

  function abscallback(darg, iarg) bind(C)
    use iso_c_binding, only : C_DOUBLE, C_INT
    real(C_DOUBLE), value :: darg
    integer(C_INT), value :: iarg
    integer(C_INT) :: abscallback
    if (darg > 0.0) then
       abscallback = iarg
    else
       abscallback = -1
    endif
  end function abscallback

!----------
  subroutine void_ptr_arg(arg0) bind(C)
    use iso_c_binding, only : C_PTR
    implicit none
    type(C_PTR), value :: arg0
    old_ptr = arg0
  end subroutine void_ptr_arg
  
!----------
  subroutine all_types(arg0, arg1, arg2, arg3, arg4, arg5) bind(C)
    use iso_c_binding, only : C_CHAR, C_INT
    implicit none
    integer(C_INT), value :: arg0
    integer(C_INT) :: arg1(*)
    character(kind=C_CHAR), value :: arg2
    character(kind=C_CHAR) :: arg3(*)
    logical(C_BOOL), value :: arg4
    logical(C_BOOL) :: arg5

    integer i

    old_int = arg0
    old_int_array(:arg0) = arg1(:arg0)
    old_char = arg2

    ! Copy CHAR array to CHARACTER variable.
    old_string = " "
    i = 1
    do while (arg3(i) .ne. C_NULL_CHAR)
       old_string(i:i) = arg3(i)
       i = i + 1
    enddo

    old_bool = arg4
    old_bool_array = arg5
  end subroutine all_types

!----------
  
end module callback_mod

program tester
  use fruit
  use iso_c_binding
  use funptr_mod
  implicit none
  real(C_DOUBLE), parameter :: pi = 3.1415926_C_DOUBLE
  integer, parameter :: lenoutbuf = 40
  logical ok

  call init_fruit

  call test_callback1
  call test_callback1_noiface
  call test_callback2
  call test_callback3
  call test_callback4
  call test_abstract_declarator
  call test_callback_arguments
  call test_return_fptr

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  ! Test passing function with interface
  subroutine test_callback1
    use callback_mod
    use state

    call set_case_name("test_callback1")

    counter = 0
    
    call callback1(incr1)
    call assert_equals(1, counter, "callback1")

    call callback1_wrap(incr1)
    call assert_equals(2, counter, "callback1_wrap")

    call callback1_external(incr1_external)
    call assert_equals(3, counter, "callback1_wrap")

    call callback1_funptr(c_funloc(incr1_funptr))
    call assert_equals(4, counter, "callback1_funptr")

  end subroutine test_callback1

  ! Test passing function without interface
  subroutine test_callback1_noiface
    use state
    external incr1_noiface

    call set_case_name("test_callback1_noiface")

    counter = 0
    
    call callback1(incr1_noiface)
    call assert_equals(1, counter, "callback1 noiface")

    call callback1_wrap(incr1_noiface)
    call assert_equals(2, counter, "callback1_wrap noiface")

    call callback1_external(incr1_noiface)
    call assert_equals(3, counter, "callback1_wrap noiface")

    call callback1_funptr(c_funloc(incr1_noiface))
    call assert_equals(4, counter, "callback1_funptr noiface")

  end subroutine test_callback1_noiface

  ! Test passing function with argument with interface
  subroutine test_callback2
    use callback_mod
    use state

    call set_case_name("test_callback2")

    counter = 0
    call callback2("one", 2, incr2)
    call assert_equals(2, counter, "callback2")

    counter = 0
    call callback2_external("two", 3, incr2)
    call assert_equals(3, counter, "callback2_wrap")

    counter = 0
    call callback2_funptr("three", 4, c_funloc(incr2))
    call assert_equals(4, counter, "callback2_funptr")

    ! call with different interface for incr

    counter = 0
    call callback2_external("double", 5, incr2_d)
    call assert_equals(5, counter, "callback2_external double")

    counter = 0
    call callback2_funptr("double", 6, c_funloc(incr2_d))
    call assert_equals(6, counter, "callback2_funptr double")

    ! call with a function instead of subroutine

    ! gfortran 12.1 assumes the same type will be passed to callback2_external.
    ! Error: Interface mismatch in dummy procedure ‘incr’ at (1):
    ! 'incr2_fun' is not a subroutine
!   counter = 0
!   call callback2_external("function", 7, incr2_fun)
!   call assert_equals(7, counter, "callback2_external function")

    counter = 0
    call callback2_funptr("function", 8, c_funloc(incr2_fun))
    call assert_equals(8, counter, "callback2_funptr function")

  end subroutine test_callback2
  
  subroutine test_callback3
    use callback_mod
    use state
    integer(C_INT) :: i_in
    real(C_DOUBLE) :: d_in

    call set_case_name("test_callback3")

    i_in = 2
    d_in = 0.0
    call callback3(1, i_in, c_funloc(incr3_int))
    call assert_equals(i_in, ival, "callback3 int ival")
    call assert_equals(d_in, dval, "callback3 int dval")

    i_in = 0
    d_in = 2.5
    call callback3(2, d_in, c_funloc(incr3_double))
    call assert_equals(i_in, ival, "callback3 int ival")
    call assert_equals(d_in, dval, "callback3 int dval")

  end subroutine test_callback3

  ! Test attributes on callback
  subroutine test_callback4
    use callback_mod
    use state
    integer(C_INT) :: rv, in(4)

    call set_case_name("test_callback4")

    in = [1,2,3,4]

    rv = callback4(in, sum4)
    call assert_equals(sum(in), rv, "callback4 sum")
    
    rv = callback4(in, product4)
    call assert_equals(product(in), rv, "callback4 product")

    ! The callback sets counter to 100
    counter = 0
    call callback_ptr(set_ptr1)
    call assert_equals(100, counter, "callback_ptr")

    dval = 0.0
    call callback_double(set_double1)
    call assert_equals(5.0d0, dval, "callback_double state")
    
  end subroutine test_callback4

  subroutine test_abstract_declarator
    use callback_mod
    integer rv

    call set_case_name("test_abstract_declarator")
    
    rv = abstract1(21, abscallback)
    call assert_equals(21, rv, "abstract2")
    
  end subroutine test_abstract_declarator

  subroutine test_callback_arguments
    ! The C function passes predefined values to the callback.
    ! The values are saved in the callback and check here.
    use callback_mod

    call set_case_name("test_callback_arguments")
    
    call callback_void_ptr(void_ptr_arg)
    call assert_false(c_associated(old_ptr), "callback_types")

    old_int = 0
    old_int_array = 0
    old_char = " "
    old_string = " "
    old_bool = .false.
    old_bool_array = .false.
    call callback_all_types(all_types)
    call assert_equals(3, old_int, "callback_all_types arg0")
    call assert_true(all([1,2,3] .eq. old_int_array(1:3)), "callback_all_types arg1")
    call assert_equals("a", old_char, "callback_all_types arg2")
    call assert_equals("dog", old_string, "callback_all_types arg3")
    call assert_true(old_bool, "callback_all_types arg4")
    call assert_true(old_bool_array, "callback_all_types arg5")
    
  end subroutine test_callback_arguments

  subroutine test_return_fptr
    type(C_FUNPTR) argptr
    procedure(pfvoid), pointer :: argfunc

    call set_case_name("test_return_fptr")

    call get_void_ptr(argptr)
    call c_f_procpointer(argptr, argfunc)
    call argfunc()
    
  end subroutine test_return_fptr
  
end program tester
