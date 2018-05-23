! wrapfvectors.f
! This is generated code, do not edit
! #######################################################################
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

        subroutine c_vector_iota_bufferify(Darg) &
                bind(C, name="VEC_vector_iota_bufferify")
            use iso_c_binding, only : C_INT
            import :: SHROUD_array
            implicit none
            type(SHROUD_array), intent(INOUT) :: Darg
        end subroutine c_vector_iota_bufferify

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
        subroutine SHROUD_array_copy_int(context, c_var, c_var_size) &
            bind(C, name="VEC_SHROUD_array_copy_int")
            use iso_c_binding, only : C_INT, C_SIZE_T
            import SHROUD_array
            type(SHROUD_array), intent(IN) :: context
            integer(C_INT), intent(OUT) :: c_var(*)
            integer(C_SIZE_T), value :: c_var_size
        end subroutine SHROUD_array_copy_int
    end interface

contains

    ! int vector_sum(const std::vector<int> & arg +dimension(:)+intent(in))
    ! arg_to_buffer
    ! function_index=0
    function vector_sum(arg) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_LONG
        integer(C_INT), intent(IN) :: arg(:)
        integer(C_INT) :: SHT_rv
        ! splicer begin function.vector_sum
        SHT_rv = c_vector_sum_bufferify(arg, size(arg, kind=C_LONG))
        ! splicer end function.vector_sum
    end function vector_sum

    ! void vector_iota(std::vector<int> & arg +dimension(:)+intent(out))
    ! arg_to_buffer
    ! function_index=1
    subroutine vector_iota(arg)
        use iso_c_binding, only : C_INT, C_SIZE_T
        integer(C_INT), intent(OUT) :: arg(:)
        type(SHROUD_array) :: Darg
        ! splicer begin function.vector_iota
        call c_vector_iota_bufferify(Darg)
        ! splicer end function.vector_iota
        call SHROUD_array_copy_int(Darg, arg, size(arg,kind=C_SIZE_T))
    end subroutine vector_iota

    ! void vector_increment(std::vector<int> & arg +dimension(:)+intent(inout))
    ! arg_to_buffer
    ! function_index=2
    subroutine vector_increment(arg)
        use iso_c_binding, only : C_INT, C_LONG, C_SIZE_T
        integer(C_INT), intent(INOUT) :: arg(:)
        type(SHROUD_array) :: Darg
        ! splicer begin function.vector_increment
        call c_vector_increment_bufferify(arg, size(arg, kind=C_LONG), &
            Darg)
        ! splicer end function.vector_increment
        call SHROUD_array_copy_int(Darg, arg, size(arg,kind=C_SIZE_T))
    end subroutine vector_increment

    ! int vector_string_count(const std::vector<std::string> & arg +dimension(:)+intent(in))
    ! arg_to_buffer
    ! function_index=3
    !>
    !! \brief count number of underscore in vector of strings
    !!
    !<
    function vector_string_count(arg) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_LONG
        character(*), intent(IN) :: arg(:)
        integer(C_INT) :: SHT_rv
        ! splicer begin function.vector_string_count
        SHT_rv = c_vector_string_count_bufferify(arg, &
            size(arg, kind=C_LONG), len(arg, kind=C_INT))
        ! splicer end function.vector_string_count
    end function vector_string_count

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module vectors_mod
