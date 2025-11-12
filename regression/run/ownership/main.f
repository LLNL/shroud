! Copyright Shroud Project Developers. See LICENSE file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from ownership.yaml.
!
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
  call test_pod_arg
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
    integer(C_INT), allocatable :: inta1(:)
!    integer(C_INT) :: lencptr
!    type(C_PTR) cptr
    type(OWN_SHROUD_capsule) cap

    call set_case_name("test_pod")
    !----------------------------------------
    ! return scalar

    ! deref(scalar)
    rv_int = return_int_ptr_scalar()
    call assert_equals(10, rv_int, "ReturnIntPtrScalar value")

    ! deref(pointer)
    nullify(intp)
    intp => return_int_ptr_pointer()
    call assert_true(associated(intp))
    call assert_equals(1, intp, "ReturnIntPtrPointe value")

    !----------------------------------------
    ! return dimension(len) owner(library)

!    cptr = return_int_ptr_dim_raw(lencptr)

    ! deref(pointer)
    nullify(intp1)
    intp1 => return_int_ptr_dim_pointer()
    call assert_true(associated(intp1))
    call assert_equals(7 , size(intp1))
    call assert_true( all(intp1 == [11,12,13,14,15,16,17]), &
         "ReturnIntPtrDimPointer value")

    inta1 = return_int_ptr_dim_alloc()
    call assert_true(allocated(inta1))
    call assert_equals(7 , size(inta1))
    call assert_true( all(inta1 == [21,22,23,24,25,26,27]), &
         "ReturnIntPtrDimAlloc value")
    deallocate(inta1)

    nullify(intp1)
    intp1 => return_int_ptr_dim_default()
    call assert_true(associated(intp1))
    call assert_equals(7 , size(intp1))
    call assert_true( all(intp1 == [31,32,33,34,35,36,37]), &
         "ReturnIntPtrDimDefault value")

    !----------------------------------------
    ! return dimension(len) owner(caller)
    ! XXX - how to delete c++ array

    ! deref(pointer)
    nullify(intp1)
    intp1 => return_int_ptr_dim_pointer_new(cap)
    call assert_true(associated(intp1))
    call assert_equals(5 , size(intp1))
    call assert_true( all(intp1 == [10,11,12,13,14]), &
         "ReturnIntPtrDimPointerNew value")
    call cap%delete

    nullify(intp1)
    intp1 => return_int_ptr_dim_default_new(cap)
    call assert_true(associated(intp1))
    call assert_equals(5 , size(intp1))
    call assert_true( all(intp1 == [30,31,32,33,34]), &
         "ReturnIntPtrDim_default_new value")
    call cap%delete()
    call SHROUD_capsule_final(cap)  ! no-op

  end subroutine test_pod

  subroutine test_pod_arg
!    integer(C_INT), pointer :: intp, intp1(:)
!    integer(C_INT), allocatable :: inta1(:)
!    integer(C_INT) :: lencptr
!    type(C_PTR) cptr
!    type(OWN_SHROUD_capsule) cap

    call set_case_name("test_pod_arg")

!    call int_ptr_dim_raw(cptr)
!   void IntPtrDimRaw(int **array, int *len+intent(out))
!  void IntPtrDimPointer(int **array, int *len+intent(out)+hidden)
!  void IntPtrDimAlloc(int **array, int *len+intent(out)+hidden)
!  void IntPtrDimDefault(int **array, int *len+intent(out)+hidden)

  end subroutine test_pod_arg

  subroutine test_class
    type(class1) obj2, obj3

    call set_case_name("test_class")

    call create_class_static(2)

    obj2 = get_class_static()
    call assert_equals(2 , obj2%get_flag(), "getClassStatic")

    obj3 = get_class_new(3)
    call assert_equals(3 , obj3%get_flag(), "getClassNew")
    call obj3%dtor

  end subroutine test_class

end program tester
