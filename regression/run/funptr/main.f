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
  integer counter
end module state


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
  subroutine incr1_int() bind(C)
    ! A bind(C) is required for the intel/oneapi/ibm compiler since
    ! it is passed directly to C via a bind(C) interface.
    use state
    implicit none
    counter = counter + 1
  end subroutine incr1_int

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
    counter = counter + i
  end subroutine incr2

!----------  

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
!  subroutine set_alloc(tc, arr) bind(C)
!    use iso_c_binding, only : C_INT
!    use funptr_mod, only : array_info
!    integer(C_INT), intent(IN), value :: tc
!    type(array_info), intent(INOUT) :: arr
!    arr%tc = tc
!  end subroutine set_alloc
  
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
  call test_callback2

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_callback1
    use callback_mod
    use state

    call set_case_name("test_callback1")

    counter = 0
    
    call callback1(incr1_int)
    call assert_equals(1, counter, "callback1")

    call callback1_wrap(incr1_int)
    call assert_equals(2, counter, "callback1_wrap")

    call callback1_external(incr1_external)
    call assert_equals(3, counter, "callback1_wrap")

    call callback1_funptr(c_funloc(incr1_funptr))
    call assert_equals(4, counter, "callback1_funptr")

  end subroutine test_callback1

  subroutine test_callback2
    use callback_mod
    use state

    call set_case_name("test_callback1")

    counter = 0
    
    call callback2("one", 2, incr2)
    call assert_equals(2, counter, "callback2")

!    call callback2_wrap(incr1_int)
!    call assert_equals(2, counter, "callback2_wrap")

!    call callback2_external(incr1_external)
!    call assert_equals(3, counter, "callback2_wrap")

!    call callback2_funptr(c_funloc(incr1_funptr))
!    call assert_equals(4, counter, "callback2_funptr")

  end subroutine test_callback2

!--  subroutine test_callback
!--    use callback_mod
!--    integer(C_INT) arg_int
!--    real(C_DOUBLE) arg_dbl
!--    character(lenoutbuf)  :: outbuf
!--!   type(array_info) :: arr
!--    external incr2_int, incr2_double
!--    external incr3_int, incr3_double
!--
!--    call set_case_name("test_callback")

    ! incr_int matches the prototype in the YAML file.
    ! incr_double, does not.

    !----------
    ! callback2, accept any type of function.
    ! first argument, tells C how to cast pointer.
!--    arg_int = 10_C_INT
!--    call callback2(1, arg_int, incr2_int)
!--    call assert_equals(30, arg_int, "incr2_int")
!--
!--    arg_dbl = 3.4_C_DOUBLE
!--    call callback2(2, arg_dbl, incr2_double)
!--    call assert_equals(23.9_C_DOUBLE, arg_dbl, "incr2_int")
!--
!--    !----------
!--    ! callback3, accept any type of function.
!--    ! first argument, tells C how to cast pointer.
!--    arg_int = 10_C_INT
!--    call callback3("int", arg_int, incr3_int, outbuf)
!--    call assert_equals(30, arg_int, "incr3_int")
!--
!--    arg_dbl = 3.4_C_DOUBLE
!--    call callback3("double", arg_dbl, incr3_double, outbuf)
!--    call assert_equals(23.9_C_DOUBLE, arg_dbl, "incr3_double")
!--
!--    !----------
!--    ! routines from a module, with an implicit interface.
!--    ! callback2, accept any type of function.
!--    ! first argument, tells C how to cast pointer.
!--    arg_int = 10_C_INT
!--    call callback2(1, arg_int, incr2b_int)
!--    call assert_equals(30, arg_int, "incr2b_int")
!--
!--    arg_dbl = 3.4_C_DOUBLE
!--    call callback2(2, arg_dbl, incr2b_double)
!--    call assert_equals(23.9_C_DOUBLE, arg_dbl, "incr2b_double")
!--
!--    !----------
!--    ! callback3, accept any type of function.
!--    ! first argument, tells C how to cast pointer.
!--    arg_int = 10_C_INT
!--    call callback3("int", arg_int, incr3b_int, outbuf)
!--    call assert_equals(30, arg_int, "incr3b_int")
!--
!--    arg_dbl = 3.4_C_DOUBLE
!--    call callback3("double", arg_dbl, incr3b_double, outbuf)
!--    call assert_equals(23.9_C_DOUBLE, arg_dbl, "incr3b_double")
!--
!--    ! The callback sets tc
!--    arr%tc = 0
!--    call callback_set_alloc(3, arr, set_alloc)
!--    call assert_equals(3, arr%tc, "callback_set_alloc")
    

!--  end subroutine test_callback

end program tester
