! wrapfvector.f
! This is generated code, do not edit
!>
!! \file wrapfvector.f
!! \brief Shroud generated wrapper for vector class
!<
! splicer begin file_top
! splicer end file_top
module vector_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.vector_0.module_use
    ! splicer end class.vector_0.module_use
    implicit none


    type, bind(C) :: SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_capsule_data

    ! splicer begin class.vector_0.module_top
    ! splicer end class.vector_0.module_top

    type vector_0
        type(SHROUD_capsule_data) :: cxxmem
        ! splicer begin class.vector_0.component_part
        ! splicer end class.vector_0.component_part
    contains
        procedure :: push_back_XXXX => vector_push_back_XXXX
        procedure :: get_instance => vector_get_instance
        procedure :: set_instance => vector_set_instance
        procedure :: associated => vector_associated
        ! splicer begin class.vector_0.type_bound_procedure_part
        ! splicer end class.vector_0.type_bound_procedure_part
    end type vector_0

    interface operator (.eq.)
        module procedure vector_0_eq
    end interface

    interface operator (.ne.)
        module procedure vector_0_ne
    end interface

    interface

        subroutine c_vector_push_back_xxxx(self, value) &
                bind(C, name="TEM_vector_push_back_XXXX")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), intent(IN) :: value
        end subroutine c_vector_push_back_xxxx

        ! splicer begin class.vector_0.additional_interfaces
        ! splicer end class.vector_0.additional_interfaces
    end interface

contains

    subroutine vector_push_back_XXXX(obj, value)
        use iso_c_binding, only : C_INT
        class(vector_0) :: obj
        integer(C_INT), intent(IN) :: value
        ! splicer begin class.vector_0.method.push_back_XXXX
        call c_vector_push_back_xxxx(obj%cxxmem, value)
        ! splicer end class.vector_0.method.push_back_XXXX
    end subroutine vector_push_back_XXXX

    ! Return pointer to C++ memory.
    function vector_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(vector_0), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function vector_get_instance

    subroutine vector_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(vector_0), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine vector_set_instance

    function vector_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(vector_0), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function vector_associated

    ! splicer begin class.vector_0.additional_functions
    ! splicer end class.vector_0.additional_functions

    function vector_0_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(vector_0), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vector_0_eq

    function vector_0_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(vector_0), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vector_0_ne

end module vector_mod
