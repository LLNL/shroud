! Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
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

  integer counter

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
  implicit none
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

  subroutine incr2(i) bind(C)
    use iso_c_binding
    use state
    implicit none
    integer(C_INT), value :: i
    counter = i
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
    
  end subroutine test_callback4

  
end program tester
