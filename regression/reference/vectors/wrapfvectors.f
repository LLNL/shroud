! wrapfvectors.f
! This is generated code, do not edit
! #######################################################################
! Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
!
! Produced at the Lawrence Livermore National Laboratory
!
! LLNL-CODE-738041.
!
! All rights reserved.
!
! This file is part of Shroud.
!
! For details about use and distribution, please read LICENSE.
!
! #######################################################################
!>
!! \file wrapfvectors.f
!! \brief Shroud generated wrapper for vectors library
!<
! splicer begin file_top
! splicer end file_top
module vectors_mod
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

    type, bind(C) :: SHROUD_array
        type(SHROUD_capsule_data) :: cxx       ! address of C++ memory
        type(C_PTR) :: addr = C_NULL_PTR       ! address of data in cxx
        integer(C_SIZE_T) :: len = 0_C_SIZE_T  ! bytes-per-item or character len of data in cxx
        integer(C_SIZE_T) :: size = 0_C_SIZE_T ! size of data in cxx
    end type SHROUD_array

    interface

        function c_vector_sum_bufferify(arg, Sarg) &
                result(SHT_rv) &
                bind(C, name="VEC_vector_sum_bufferify")
            use iso_c_binding, only : C_INT, C_LONG
            implicit none
            integer(C_INT), intent(IN) :: arg(*)
            integer(C_LONG), value, intent(IN) :: Sarg
            integer(C_INT) :: SHT_rv
        end function c_vector_sum_bufferify

        subroutine c_vector_iota_out_bufferify(Darg) &
                bind(C, name="VEC_vector_iota_out_bufferify")
            use iso_c_binding, only : C_INT
            import :: SHROUD_array
            implicit none
            type(SHROUD_array), intent(INOUT) :: Darg
        end subroutine c_vector_iota_out_bufferify

        subroutine c_vector_iota_out_alloc_bufferify(Darg) &
                bind(C, name="VEC_vector_iota_out_alloc_bufferify")
            use iso_c_binding, only : C_INT
            import :: SHROUD_array
            implicit none
            type(SHROUD_array), intent(INOUT) :: Darg
        end subroutine c_vector_iota_out_alloc_bufferify

        subroutine c_vector_iota_inout_alloc_bufferify(arg, Sarg, Darg) &
                bind(C, name="VEC_vector_iota_inout_alloc_bufferify")
            use iso_c_binding, only : C_INT, C_LONG
            import :: SHROUD_array
            implicit none
            integer(C_INT), intent(INOUT) :: arg(*)
            integer(C_LONG), value, intent(IN) :: Sarg
            type(SHROUD_array), intent(INOUT) :: Darg
        end subroutine c_vector_iota_inout_alloc_bufferify

        subroutine c_vector_increment_bufferify(arg, Sarg, Darg) &
                bind(C, name="VEC_vector_increment_bufferify")
            use iso_c_binding, only : C_INT, C_LONG
            import :: SHROUD_array
            implicit none
            integer(C_INT), intent(INOUT) :: arg(*)
            integer(C_LONG), value, intent(IN) :: Sarg
            type(SHROUD_array), intent(INOUT) :: Darg
        end subroutine c_vector_increment_bufferify

        function c_vector_string_count_bufferify(arg, Sarg, Narg) &
                result(SHT_rv) &
                bind(C, name="VEC_vector_string_count_bufferify")
            use iso_c_binding, only : C_CHAR, C_INT, C_LONG
            implicit none
            character(kind=C_CHAR), intent(IN) :: arg(*)
            integer(C_LONG), value, intent(IN) :: Sarg
            integer(C_INT), value, intent(IN) :: Narg
            integer(C_INT) :: SHT_rv
        end function c_vector_string_count_bufferify

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

    interface
        ! helper function
        ! Copy contents of context into c_var.
        subroutine SHROUD_copy_array_int(context, c_var, c_var_size) &
            bind(C, name="VEC_ShroudCopyArray")
            use iso_c_binding, only : C_INT, C_SIZE_T
            import SHROUD_array
            type(SHROUD_array), intent(IN) :: context
            integer(C_INT), intent(OUT) :: c_var(*)
            integer(C_SIZE_T), value :: c_var_size
        end subroutine SHROUD_copy_array_int
    end interface

contains

    ! int vector_sum(const std::vector<int> & arg +dimension(:)+intent(in))
    ! arg_to_buffer
    function vector_sum(arg) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_LONG
        integer(C_INT), intent(IN) :: arg(:)
        integer(C_INT) :: SHT_rv
        ! splicer begin function.vector_sum
        SHT_rv = c_vector_sum_bufferify(arg, size(arg, kind=C_LONG))
        ! splicer end function.vector_sum
    end function vector_sum

    ! void vector_iota_out(std::vector<int> & arg +dimension(:)+intent(out))
    ! arg_to_buffer
    !>
    !! \brief Copy vector into Fortran input array
    !!
    !<
    subroutine vector_iota_out(arg)
        use iso_c_binding, only : C_INT, C_SIZE_T
        integer(C_INT), intent(OUT) :: arg(:)
        type(SHROUD_array) :: Darg
        ! splicer begin function.vector_iota_out
        call c_vector_iota_out_bufferify(Darg)
        ! splicer end function.vector_iota_out
        call SHROUD_copy_array_int(Darg, arg, size(arg,kind=C_SIZE_T))
    end subroutine vector_iota_out

    ! void vector_iota_out_alloc(std::vector<int> & arg +deref(allocatable)+dimension(:)+intent(out))
    ! arg_to_buffer
    !>
    !! \brief Copy vector into Fortran allocatable array
    !!
    !<
    subroutine vector_iota_out_alloc(arg)
        use iso_c_binding, only : C_INT, C_SIZE_T
        integer(C_INT), intent(OUT), allocatable :: arg(:)
        type(SHROUD_array) :: Darg
        ! splicer begin function.vector_iota_out_alloc
        call c_vector_iota_out_alloc_bufferify(Darg)
        ! splicer end function.vector_iota_out_alloc
        allocate(arg(Darg%size))
        call SHROUD_copy_array_int(Darg, arg, size(arg,kind=C_SIZE_T))
    end subroutine vector_iota_out_alloc

    ! void vector_iota_inout_alloc(std::vector<int> & arg +deref(allocatable)+dimension(:)+intent(inout))
    ! arg_to_buffer
    !>
    !! \brief Copy vector into Fortran allocatable array
    !!
    !<
    subroutine vector_iota_inout_alloc(arg)
        use iso_c_binding, only : C_INT, C_LONG, C_SIZE_T
        integer(C_INT), intent(INOUT), allocatable :: arg(:)
        type(SHROUD_array) :: Darg
        ! splicer begin function.vector_iota_inout_alloc
        call c_vector_iota_inout_alloc_bufferify(arg, &
            size(arg, kind=C_LONG), Darg)
        ! splicer end function.vector_iota_inout_alloc
        if (allocated(arg)) deallocate(arg)
        allocate(arg(Darg%size))
        call SHROUD_copy_array_int(Darg, arg, size(arg,kind=C_SIZE_T))
    end subroutine vector_iota_inout_alloc

    ! void vector_increment(std::vector<int> & arg +dimension(:)+intent(inout))
    ! arg_to_buffer
    subroutine vector_increment(arg)
        use iso_c_binding, only : C_INT, C_LONG, C_SIZE_T
        integer(C_INT), intent(INOUT) :: arg(:)
        type(SHROUD_array) :: Darg
        ! splicer begin function.vector_increment
        call c_vector_increment_bufferify(arg, size(arg, kind=C_LONG), &
            Darg)
        ! splicer end function.vector_increment
        call SHROUD_copy_array_int(Darg, arg, size(arg,kind=C_SIZE_T))
    end subroutine vector_increment

    ! int vector_string_count(const std::vector<std::string> & arg +dimension(:)+intent(in))
    ! arg_to_buffer
    !>
    !! \brief count number of underscore in vector of strings
    !!
    !<
    function vector_string_count(arg) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_LONG
        character(len=*), intent(IN) :: arg(:)
        integer(C_INT) :: SHT_rv
        ! splicer begin function.vector_string_count
        SHT_rv = c_vector_string_count_bufferify(arg, &
            size(arg, kind=C_LONG), len(arg, kind=C_INT))
        ! splicer end function.vector_string_count
    end function vector_string_count

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module vectors_mod
