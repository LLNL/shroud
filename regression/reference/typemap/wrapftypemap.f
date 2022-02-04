! wrapftypemap.f
! This file is generated by Shroud nowrite-version. Do not edit.
! Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
! other Shroud Project Developers.
! See the top-level COPYRIGHT file for details.
!
! SPDX-License-Identifier: (BSD-3-Clause)
!
!>
!! \file wrapftypemap.f
!! \brief Shroud generated wrapper for typemap library
!<
! splicer begin file_top
! splicer end file_top
module typemap_mod
    use iso_c_binding, only : C_DOUBLE, C_FLOAT, C_INT32_T, C_INT64_T
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
#if defined(USE_64BIT_INDEXTYPE)
    integer, parameter :: INDEXTYPE = C_INT64_T
#else
    integer, parameter :: INDEXTYPE = C_INT32_T
#endif

#if defined(USE_64BIT_FLOAT)
    integer, parameter :: FLOATTYPE = C_DOUBLE
#else
    integer, parameter :: FLOATTYPE = C_FLOAT
#endif
    ! splicer end module_top

    interface

        function c_pass_index(i1, i2) &
                result(SHT_rv) &
                bind(C, name="TYP_pass_index")
            use iso_c_binding, only : C_BOOL
            import :: INDEXTYPE
            implicit none
            integer(INDEXTYPE), value, intent(IN) :: i1
            integer(INDEXTYPE), intent(OUT) :: i2
            logical(C_BOOL) :: SHT_rv
        end function c_pass_index

        subroutine pass_float(f1) &
                bind(C, name="TYP_pass_float")
            import :: FLOATTYPE
            implicit none
            real(FLOATTYPE), value, intent(IN) :: f1
        end subroutine pass_float

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

contains

    function pass_index(i1, i2) &
            result(SHT_rv)
        use iso_c_binding, only : C_BOOL
        integer(INDEXTYPE), value, intent(IN) :: i1
        integer(INDEXTYPE), intent(OUT) :: i2
        logical :: SHT_rv
        ! splicer begin function.pass_index
        SHT_rv = c_pass_index(i1, i2)
        ! splicer end function.pass_index
    end function pass_index

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module typemap_mod
