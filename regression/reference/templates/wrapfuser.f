! wrapfuser.f
! This is generated code, do not edit
!>
!! \file wrapfuser.f
!! \brief Shroud generated wrapper for user class
!<
! splicer begin file_top
! splicer end file_top
module user_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.user_0.module_use
    ! splicer end class.user_0.module_use
    implicit none


    type, bind(C) :: SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_capsule_data

    ! splicer begin class.user_0.module_top
    ! splicer end class.user_0.module_top

    type user_0
        type(SHROUD_capsule_data) :: cxxmem
        ! splicer begin class.user_0.component_part
        ! splicer end class.user_0.component_part
    contains
        procedure :: nested_double => user_nested_double
        procedure :: get_instance => user_get_instance
        procedure :: set_instance => user_set_instance
        procedure :: associated => user_associated
        ! splicer begin class.user_0.type_bound_procedure_part
        ! splicer end class.user_0.type_bound_procedure_part
    end type user_0

    interface operator (.eq.)
        module procedure user_0_eq
    end interface

    interface operator (.ne.)
        module procedure user_0_ne
    end interface

    interface

        subroutine c_user_nested_double(self, value, arg2) &
                bind(C, name="TEM_user_nested_double")
            use iso_c_binding, only : C_DOUBLE, C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), intent(IN) :: value
            real(C_DOUBLE), value, intent(IN) :: arg2
        end subroutine c_user_nested_double

        ! splicer begin class.user_0.additional_interfaces
        ! splicer end class.user_0.additional_interfaces
    end interface

contains

    subroutine user_nested_double(obj, value, arg2)
        use iso_c_binding, only : C_DOUBLE, C_INT
        class(user_0) :: obj
        integer(C_INT), intent(IN) :: value
        real(C_DOUBLE), value, intent(IN) :: arg2
        ! splicer begin class.user_0.method.nested_double
        call c_user_nested_double(obj%cxxmem, value, arg2)
        ! splicer end class.user_0.method.nested_double
    end subroutine user_nested_double

    ! Return pointer to C++ memory.
    function user_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(user_0), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function user_get_instance

    subroutine user_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(user_0), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine user_set_instance

    function user_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(user_0), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function user_associated

    ! splicer begin class.user_0.additional_functions
    ! splicer end class.user_0.additional_functions

    function user_0_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(user_0), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function user_0_eq

    function user_0_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(user_0), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function user_0_ne

end module user_mod
