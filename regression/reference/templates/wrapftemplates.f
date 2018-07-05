! wrapftemplates.f
! This is generated code, do not edit
!>
!! \file wrapftemplates.f
!! \brief Shroud generated wrapper for templates library
!<
! splicer begin file_top
! splicer end file_top
module templates_mod
    ! splicer begin module_use
    ! splicer end module_use
    implicit none

    ! splicer begin module_top
    ! splicer end module_top

    interface

        subroutine c_function_tu_0(arg1, arg2) &
                bind(C, name="TEM_function_tu_0")
            use iso_c_binding, only : C_INT, C_LONG
            implicit none
            integer(C_INT), value, intent(IN) :: arg1
            integer(C_LONG), value, intent(IN) :: arg2
        end subroutine c_function_tu_0

        subroutine c_function_tu_1(arg1, arg2) &
                bind(C, name="TEM_function_tu_1")
            use iso_c_binding, only : C_DOUBLE, C_FLOAT
            implicit none
            real(C_FLOAT), value, intent(IN) :: arg1
            real(C_DOUBLE), value, intent(IN) :: arg2
        end subroutine c_function_tu_1

        ! splicer begin additional_interfaces
        ! splicer end additional_interfaces
    end interface

    interface function_tu
        module procedure function_tu_0
        module procedure function_tu_1
    end interface function_tu

contains

    subroutine function_tu_0(arg1, arg2)
        use iso_c_binding, only : C_INT, C_LONG
        integer(C_INT), value, intent(IN) :: arg1
        integer(C_LONG), value, intent(IN) :: arg2
        ! splicer begin function.function_tu_0
        call c_function_tu_0(arg1, arg2)
        ! splicer end function.function_tu_0
    end subroutine function_tu_0

    subroutine function_tu_1(arg1, arg2)
        use iso_c_binding, only : C_DOUBLE, C_FLOAT
        real(C_FLOAT), value, intent(IN) :: arg1
        real(C_DOUBLE), value, intent(IN) :: arg2
        ! splicer begin function.function_tu_1
        call c_function_tu_1(arg1, arg2)
        ! splicer end function.function_tu_1
    end subroutine function_tu_1

    ! splicer begin additional_functions
    ! splicer end additional_functions

end module templates_mod
