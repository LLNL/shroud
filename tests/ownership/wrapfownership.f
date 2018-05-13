! wrapfownership.f
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
!! \file wrapfownership.f
!! \brief Shroud generated wrapper for ownership library
!<
! splicer begin file_top
! splicer end file_top
module ownership_mod
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    interface

        function c_return_int_ptr() &
                result(SHT_rv) &
                bind(C, name="OWN_return_int_ptr")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            type(C_PTR) SHT_rv
        end function c_return_int_ptr

        function return_int_ptr_scalar() &
                result(SHT_rv) &
                bind(C, name="OWN_return_int_ptr_scalar")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT) :: SHT_rv
        end function return_int_ptr_scalar

        function c_return_int_ptr_dim(len) &
                result(SHT_rv) &
                bind(C, name="OWN_return_int_ptr_dim")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            integer(C_INT), intent(OUT) :: len
            type(C_PTR) SHT_rv
        end function c_return_int_ptr_dim

        function c_return_int_ptr_dim_new(len) &
                result(SHT_rv) &
                bind(C, name="OWN_return_int_ptr_dim_new")
            use iso_c_binding, only : C_INT, C_PTR
            implicit none
            integer(C_INT), intent(OUT) :: len
            type(C_PTR) SHT_rv
        end function c_return_int_ptr_dim_new

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

contains

    ! int * ReturnIntPtr()
    ! function_index=0
    function return_int_ptr() &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR, c_f_pointer
        integer(C_INT), pointer :: SHT_rv
        type(C_PTR) :: SHT_ptr
        ! splicer begin function.return_int_ptr
        SHT_ptr = c_return_int_ptr()
        call c_f_pointer(SHT_ptr, SHT_rv)
        ! splicer end function.return_int_ptr
    end function return_int_ptr

    ! int * ReturnIntPtrDim(int * len +hidden+intent(out)) +dimension(len)
    ! function_index=2
    function return_int_ptr_dim() &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR, c_f_pointer
        integer(C_INT) :: len
        integer(C_INT), pointer :: SHT_rv(:)
        type(C_PTR) :: SHT_ptr
        ! splicer begin function.return_int_ptr_dim
        SHT_ptr = c_return_int_ptr_dim(len)
        call c_f_pointer(SHT_ptr, SHT_rv, [len])
        ! splicer end function.return_int_ptr_dim
    end function return_int_ptr_dim

    ! int * ReturnIntPtrDimNew(int * len +hidden+intent(out)) +dimension(len)
    ! function_index=3
    function return_int_ptr_dim_new() &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR, c_f_pointer
        integer(C_INT) :: len
        integer(C_INT), pointer :: SHT_rv(:)
        type(C_PTR) :: SHT_ptr
        ! splicer begin function.return_int_ptr_dim_new
        SHT_ptr = c_return_int_ptr_dim_new(len)
        call c_f_pointer(SHT_ptr, SHT_rv, [len])
        ! splicer end function.return_int_ptr_dim_new
    end function return_int_ptr_dim_new

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module ownership_mod
