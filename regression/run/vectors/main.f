! Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
! #######################################################################
!
! Test Fortran API generated from vectors.yaml.
!
program tester

  use fruit
  use iso_c_binding
  use vectors_mod
  implicit none

  logical ok

  call init_fruit

  call test_vector_int
  call test_vector_double
  call test_vector_double_ptr
  call test_vector_string
  call test_return
  call test_implied

  call fruit_summary
  call fruit_finalize

  call is_all_successful(ok)
  if (.not. ok) then
     call exit(1)
  endif

contains

  subroutine test_vector_int
    integer(C_INT) intv(5), intv2(10)
    integer(C_INT), allocatable :: inta(:)
    integer irv
    integer(C_LONG) num

    call set_case_name("test_vector_int")

    intv = [1,2,3,4,5]
    irv = vector_sum(intv)
    call assert_true(irv .eq. 15, "vector_sum")

    intv(:) = 0
    call vector_iota_out(intv)
    call assert_true(all(intv(:) .eq. [1,2,3,4,5]), "vector_iota_out values")

    ! Fortran and C wrappers have custom statements.
    intv2(:) = 0
    num = vector_iota_out_with_num(intv2)
    call assert_true(5 == num, "vector_iota_out_with_num size")
    call assert_true(all(intv2(:num) .eq. [1,2,3,4,5]), &
         "vector_iota_out_with_num values")

    ! Only Fortran wrapper has custom statements.
    intv2(:) = 0
    num = vector_iota_out_with_num2(intv2)
    call assert_true(5 == num, "vector_iota_out_with_num2 size")
    call assert_true(all(intv2(:num) .eq. [1,2,3,4,5]), &
         "vector_iota_out_with_num2 values")

    ! inta is intent(out), so it will be deallocated upon entry
    ! to vector_iota_out_alloc.
    call vector_iota_out_alloc(inta)
    call assert_true(allocated(inta), "vector_iota_out_alloc")
    call assert_equals(5 , size(inta), "vector_iota_out_alloc size")
    call assert_true( all(inta == [1,2,3,4,5]), &
         "vector_iota_out_alloc value")

    ! inta is intent(inout), so it will NOT be deallocated upon entry
    ! to vector_iota_inout_alloc.
    ! Use previous value to append
    call vector_iota_inout_alloc(inta)
    call assert_true(allocated(inta), "vector_iota_inout_alloc allocated")
    call assert_equals(10 , size(inta), "vector_iota_inout_alloc size")
    call assert_true( all(inta == [1,2,3,4,5,11,12,13,14,15]), &
         "vector_iota_inout_alloc value")
    deallocate(inta)

    intv = [1,2,3,4,5]
    call vector_increment(intv)
    call assert_true(all(intv(:) .eq. [2,3,4,5,6]), &
         "vector_increment values")
  end subroutine test_vector_int

  subroutine test_vector_double
    real(C_DOUBLE) intv(5)
!    real(C_DOUBLE), allocatable :: inta(:)
!    integer irv

    call set_case_name("test_vector_double")

!    intv = [1.0, 2.0, 3.0, 4.0, 5.0]
!    irv = vector_sum(intv)
!    call assert_true(irv .eq. 15)

    intv(:) = 0
    call vector_iota_out_d(intv)
    call assert_true(all(intv(:) .eq. [1.,2.,3.,4.,5.]), &
         "vector_iota_out_d values")

    ! inta is intent(out), so it will be deallocated upon entry to vector_iota_out_alloc
!    call vector_iota_out_alloc(inta)
!    call assert_true(allocated(inta))
!    call assert_equals(5 , size(inta))
!    call assert_true( all(inta == [1,2,3,4,5]), &
!         "vector_iota_out_alloc value")

    ! inta is intent(inout), so it will NOT be deallocated upon entry to vector_iota_inout_alloc
    ! Use previous value to append
!    call vector_iota_inout_alloc(inta)
!    call assert_true(allocated(inta))
!    call assert_equals(10 , size(inta))
!    call assert_true( all(inta == [1,2,3,4,5,11,12,13,14,15]), &
!         "vector_iota_inout_alloc value")
!    deallocate(inta)

!    intv = [1,2,3,4,5]
!    call vector_increment(intv)
!    call assert_true(all(intv(:) .eq. [2,3,4,5,6]))
  end subroutine test_vector_double

  subroutine test_vector_double_ptr
    integer(C_INT) rvsum
    real(C_DOUBLE) datain(3,2)

    call set_case_name("test_vector_double_ptr")

    datain = reshape([1,2,3,4,5,6], shape(datain))

    rvsum = vector_of_pointers(datain, size(datain, 1))
    call assert_true(sum(datain) == rvsum, "vector_of_pointers")

  end subroutine test_vector_double_ptr
  
  subroutine test_vector_string
    integer irv
    character(10) :: names(3)
    character(:), allocatable :: anames(:)
    character(20), allocatable :: a20names(:)

    call set_case_name("test_vector_string")

    ! count number of underscores
    names = [ "dog_cat   ", "bird_mouse", "__        " ]
    irv = vector_string_count(names)
    call assert_true(irv == 4, "vector_string_count")

    ! Fill strings into names
    names = " "
    call vector_string_fill(names)
!    call assert_true(irv == 2)
    call assert_true( names(1) == "dog", "vector_string_fill(1)")
    call assert_true( names(2) == "bird", "vector_string_fill(2)")
    call assert_true( names(3) == " ", "vector_string_fill(3)")

    ! Fill strings into names
    call assert_false(allocated(anames), "anames not allocated")
    call vector_string_fill_allocatable(anames)
    call assert_true(allocated(anames), "anames is allocated")
    call assert_equals(2, size(anames), "size of anames")
    call assert_equals(4, len(anames), "len of anames")
    call assert_true( anames(1) == "dog", "vector_string_fill_allocatable(1)")
    call assert_true( anames(2) == "bird", "vector_string_fill_allocatable(2)")

    ! Fill strings into names with len=20
    call assert_false(allocated(a20names), "a20names not allocated")
    call vector_string_fill_allocatable_len(a20names)
    call assert_true(allocated(a20names), "a20names is allocated")
    call assert_equals(2, size(a20names), "size of a20names")
    call assert_equals(20, len(a20names), "len of a20names")
    call assert_true( a20names(1) == "dog", "vector_string_fill_allocatable(1)")
    call assert_true( a20names(2) == "bird", "vector_string_fill_allocatable(2)")

    ! Append -like to names.
    ! Note that strings will be truncated to len(names)
    names = [ "fish      ", "toolong   ", "          " ]
!    call vector_string_append(names)
!    call assert_true( names(1) == "fish-like")
!    call assert_true( names(2) == "toolong-li")
!    call assert_true( names(3) == "-like")
 
  end subroutine test_vector_string

  ! Test returning a vector as a function result
  subroutine test_return

    integer(C_INT), allocatable :: rv1(:)

    rv1 = return_vector_alloc(10)
    call assert_true(allocated(rv1), &
         "ReturnVectorAlloc allocated")
    call assert_equals(10, size(rv1), &
         "ReturnVectorAlloc size")
    call assert_true(all(rv1(:) .eq. [1,2,3,4,5,6,7,8,9,10]), &
         "ReturnVectorAlloc values")
    
  end subroutine test_return

  subroutine test_implied
    integer(C_INT) :: array2d(2,3)
    integer irv

    call set_case_name("test_implied")

    irv = return_dim2(array2d)
    call assert_equals(3, irv, "return_dim2")
    
  end subroutine test_implied

end program tester
