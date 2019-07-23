! wrapfmemdoc.f
! This is generated code, do not edit
! Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapfmemdoc.f
!! \brief Shroud generated wrapper for memdoc library
!<
! splicer begin file_top
! splicer end file_top
module memdoc_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR, C_SIZE_T
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    type, bind(C) :: SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_capsule_data

    ! start array_context
    type, bind(C) :: SHROUD_array
        type(SHROUD_capsule_data) :: cxx       ! address of C++ memory
        type(C_PTR) :: addr = C_NULL_PTR       ! address of data in cxx
        integer(C_SIZE_T) :: len = 0_C_SIZE_T  ! bytes-per-item or character len of data in cxx
        integer(C_SIZE_T) :: size = 0_C_SIZE_T ! size of data in cxx
    end type SHROUD_array
    ! end array_context

    interface

        ! start c_get_const_string_ptr_alloc
        function c_get_const_string_ptr_alloc() &
                result(SHT_rv) &
                bind(C, name="STR_get_const_string_ptr_alloc")
            use iso_c_binding, only : C_PTR
            implicit none
            type(C_PTR) SHT_rv
        end function c_get_const_string_ptr_alloc
        ! end c_get_const_string_ptr_alloc

        ! start c_get_const_string_ptr_alloc_bufferify
        subroutine c_get_const_string_ptr_alloc_bufferify(DSHF_rv) &
                bind(C, name="STR_get_const_string_ptr_alloc_bufferify")
            import :: SHROUD_array
            implicit none
            type(SHROUD_array), intent(INOUT) :: DSHF_rv
        end subroutine c_get_const_string_ptr_alloc_bufferify
        ! end c_get_const_string_ptr_alloc_bufferify

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

    interface
        ! helper function
        ! Copy the char* or std::string in context into c_var.
        subroutine SHROUD_copy_string_and_free(context, c_var, c_var_size) &
             bind(c,name="STR_ShroudCopyStringAndFree")
            use, intrinsic :: iso_c_binding, only : C_CHAR, C_SIZE_T
            import SHROUD_array
            type(SHROUD_array), intent(IN) :: context
            character(kind=C_CHAR), intent(OUT) :: c_var(*)
            integer(C_SIZE_T), value :: c_var_size
        end subroutine SHROUD_copy_string_and_free
    end interface

contains

    ! start get_const_string_ptr_alloc
    function get_const_string_ptr_alloc() &
            result(SHT_rv)
        type(SHROUD_array) :: DSHF_rv
        character(len=:), allocatable :: SHT_rv
        ! splicer begin function.get_const_string_ptr_alloc
        call c_get_const_string_ptr_alloc_bufferify(DSHF_rv)
        ! splicer end function.get_const_string_ptr_alloc
        allocate(character(len=DSHF_rv%len):: SHT_rv)
        call SHROUD_copy_string_and_free(DSHF_rv, SHT_rv, DSHF_rv%len)
    end function get_const_string_ptr_alloc
    ! end get_const_string_ptr_alloc

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module memdoc_mod
