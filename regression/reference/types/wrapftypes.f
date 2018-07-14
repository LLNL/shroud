! wrapftypes.f
! This is generated code, do not edit
! #######################################################################
! Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
!! \file wrapftypes.f
!! \brief Shroud generated wrapper for types library
!<
! splicer begin file_top
! splicer end file_top
module types_mod
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    interface

        function short_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_short_func")
            use iso_c_binding, only : C_SHORT
            implicit none
            integer(C_SHORT), value, intent(IN) :: arg1
            integer(C_SHORT) :: SHT_rv
        end function short_func

        function int_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_int_func")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: arg1
            integer(C_INT) :: SHT_rv
        end function int_func

        function long_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_long_func")
            use iso_c_binding, only : C_LONG
            implicit none
            integer(C_LONG), value, intent(IN) :: arg1
            integer(C_LONG) :: SHT_rv
        end function long_func

        function long2_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_long2_func")
            use iso_c_binding, only : C_LONG_LONG
            implicit none
            integer(C_LONG_LONG), value, intent(IN) :: arg1
            integer(C_LONG_LONG) :: SHT_rv
        end function long2_func

        function ushort_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_ushort_func")
            use iso_c_binding, only : C_SHORT
            implicit none
            integer(C_SHORT), value, intent(IN) :: arg1
            integer(C_SHORT) :: SHT_rv
        end function ushort_func

        function uint_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_uint_func")
            use iso_c_binding, only : C_INT
            implicit none
            integer(C_INT), value, intent(IN) :: arg1
            integer(C_INT) :: SHT_rv
        end function uint_func

        function ulong_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_ulong_func")
            use iso_c_binding, only : C_LONG
            implicit none
            integer(C_LONG), value, intent(IN) :: arg1
            integer(C_LONG) :: SHT_rv
        end function ulong_func

        function ulong_long_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_ulong_long_func")
            use iso_c_binding, only : C_LONG_LONG
            implicit none
            integer(C_LONG_LONG), value, intent(IN) :: arg1
            integer(C_LONG_LONG) :: SHT_rv
        end function ulong_long_func

        function int8_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_int8_func")
            use iso_c_binding, only : C_INT8_T
            implicit none
            integer(C_INT8_T), value, intent(IN) :: arg1
            integer(C_INT8_T) :: SHT_rv
        end function int8_func

        function int16_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_int16_func")
            use iso_c_binding, only : C_INT16_T
            implicit none
            integer(C_INT16_T), value, intent(IN) :: arg1
            integer(C_INT16_T) :: SHT_rv
        end function int16_func

        function int32_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_int32_func")
            use iso_c_binding, only : C_INT32_T
            implicit none
            integer(C_INT32_T), value, intent(IN) :: arg1
            integer(C_INT32_T) :: SHT_rv
        end function int32_func

        function int64_func(arg1) &
                result(SHT_rv) &
                bind(C, name="TYP_int64_func")
            use iso_c_binding, only : C_INT64_T
            implicit none
            integer(C_INT64_T), value, intent(IN) :: arg1
            integer(C_INT64_T) :: SHT_rv
        end function int64_func

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

contains

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module types_mod
