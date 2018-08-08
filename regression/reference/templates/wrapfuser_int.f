! wrapfuser_int.f
! This is generated code, do not edit
!>
!! \file wrapfuser_int.f
!! \brief Shroud generated wrapper for user class
!<
! splicer begin file_top
! splicer end file_top
module user_int_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.user_int.module_use
    ! splicer end class.user_int.module_use
    implicit none


    ! splicer begin class.user_int.module_top
    ! splicer end class.user_int.module_top

    type, bind(C) :: SHROUD_user_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_user_capsule

    type user_int
        type(SHROUD_user_capsule) :: cxxmem
        ! splicer begin class.user_int.component_part
        ! splicer end class.user_int.component_part
    contains
        procedure :: nested_double => user_int_nested_double
        procedure :: get_instance => user_int_get_instance
        procedure :: set_instance => user_int_set_instance
        procedure :: associated => user_int_associated
        ! splicer begin class.user_int.type_bound_procedure_part
        ! splicer end class.user_int.type_bound_procedure_part
    end type user_int

    interface operator (.eq.)
        module procedure user_int_eq
    end interface

    interface operator (.ne.)
        module procedure user_int_ne
    end interface

    interface

        subroutine c_user_int_nested_double(self, arg1, arg2) &
                bind(C, name="TEM_user_int_nested_double")
            use iso_c_binding, only : C_DOUBLE, C_INT
            import :: SHROUD_user_capsule
            implicit none
            type(SHROUD_user_capsule), intent(IN) :: self
            integer(C_INT), value, intent(IN) :: arg1
            real(C_DOUBLE), value, intent(IN) :: arg2
        end subroutine c_user_int_nested_double

        ! splicer begin class.user_int.additional_interfaces
        ! splicer end class.user_int.additional_interfaces
    end interface

contains

    subroutine user_int_nested_double(obj, arg1, arg2)
        use iso_c_binding, only : C_DOUBLE, C_INT
        class(user_int) :: obj
        integer(C_INT), value, intent(IN) :: arg1
        real(C_DOUBLE), value, intent(IN) :: arg2
        ! splicer begin class.user_int.method.nested_double
        call c_user_int_nested_double(obj%cxxmem, arg1, arg2)
        ! splicer end class.user_int.method.nested_double
    end subroutine user_int_nested_double

    ! Return pointer to C++ memory.
    function user_int_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(user_int), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function user_int_get_instance

    subroutine user_int_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(user_int), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine user_int_set_instance

    function user_int_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(user_int), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function user_int_associated

    ! splicer begin class.user_int.additional_functions
    ! splicer end class.user_int.additional_functions

    function user_int_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(user_int), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function user_int_eq

    function user_int_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(user_int), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function user_int_ne

end module user_int_mod
