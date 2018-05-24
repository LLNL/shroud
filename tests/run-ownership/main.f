! Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC. 
! Produced at the Lawrence Livermore National Laboratory 
!
! LLNL-CODE-738041.
! All rights reserved. 
!
! This file is part of Shroud.  For details, see
! https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are
! met:
!
! * Redistributions of source code must retain the above copyright
!   notice, this list of conditions and the disclaimer below.
! 
! * Redistributions in binary form must reproduce the above copyright
!   notice, this list of conditions and the disclaimer (as noted below)
!   in the documentation and/or other materials provided with the
!   distribution.
!
! * Neither the name of the LLNS/LLNL nor the names of its contributors
!   may be used to endorse or promote products derived from this
!   software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
! A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
! LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
! CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
! EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
! PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
! PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
! LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
! NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
! SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
!
! #######################################################################

program tester
  use fruit
  use iso_c_binding
  use ownership_mod
  implicit none
  logical ok

!  logical rv_logical, wrk_logical
!  integer rv_integer
  integer(C_INT) rv_int
!  real(C_DOUBLE) rv_double
!  character(30) rv_char

  call init_fruit

  call test_pod
  call test_class

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_pod
    integer(C_INT), pointer :: intp, intp1(:)
!    integer(C_INT), allocatable :: inta1(:)

    !----------------------------------------
    ! return scalar

    ! deref(scalar)
    rv_int = return_int_ptr_scalar()
    call assert_equals(10, rv_int, "return_int_ptr_scalar value")

    ! deref(pointer)
    nullify(intp)
    intp => return_int_ptr_pointer()
    call assert_true(associated(intp))
    call assert_equals(1, intp, "return_int_ptr value")

    !----------------------------------------
    ! return dimension(len) owner(library)

    ! deref(pointer)
    nullify(intp1)
    intp1 => return_int_ptr_dim_pointer()
    call assert_true(associated(intp1))
    call assert_equals(7 , size(intp1))
    call assert_true( all(intp1 == [11,12,13,14,15,16,17]), &
         "return_int_ptr_dim_pointer value")

!    deallocate(inta1)
!    inta1 = return_int_ptr_dim_alloc()
!    call assert_true(allocated(inta1))
!    call assert_equals(7 , size(inta1))
!    call assert_true( all(intp1 == [21,22,23,24,25,26,27]), &
!         "return_int_ptr_dim_alloc value")

    nullify(intp1)
    intp1 => return_int_ptr_dim_default()
    call assert_true(associated(intp1))
    call assert_equals(7 , size(intp1))
    call assert_true( all(intp1 == [31,32,33,34,35,36,37]), &
         "return_int_ptr_dim_default value")

    !----------------------------------------
    ! return dimension(len) owner(caller)
    ! XXX - how to delete c++ array

    ! deref(pointer)
    nullify(intp1)
    intp1 => return_int_ptr_dim_pointer_new()
    call assert_true(associated(intp1))
    call assert_equals(5 , size(intp1))
    call assert_true( all(intp1 == [10,11,12,13,14]), &
         "return_int_ptr_dim_pointer_new value")

    nullify(intp1)
    intp1 => return_int_ptr_dim_default_new()
    call assert_true(associated(intp1))
    call assert_equals(5 , size(intp1))
    call assert_true( all(intp1 == [30,31,32,33,34]), &
         "return_int_ptr_dim_default_new value")

  end subroutine test_pod

  subroutine test_class
    type(class1) obj2, obj3

    call create_class_static(2)

    obj2 = get_class_static()
    call assert_equals(2 , obj2%get_flag())

    obj3 = get_class_new(3)
    call assert_equals(3 , obj3%get_flag())

  end subroutine test_class

end program tester
