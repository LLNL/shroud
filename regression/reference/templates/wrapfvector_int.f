! wrapfvector_int.f
! This is generated code, do not edit
!>
!! \file wrapfvector_int.f
!! \brief Shroud generated wrapper for vector class
!<
! splicer begin file_top
! splicer end file_top
module vector_int_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.vector_int.module_use
    ! splicer end class.vector_int.module_use
    implicit none


    type, bind(C) :: SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_capsule_data

    ! splicer begin class.vector_int.module_top
    ! splicer end class.vector_int.module_top

    type vector_int
        type(SHROUD_capsule_data) :: cxxmem
        ! splicer begin class.vector_int.component_part
        ! splicer end class.vector_int.component_part
    contains
        procedure :: dtor => vector_int_dtor
        procedure :: push_back => vector_int_push_back
        procedure :: get_instance => vector_int_get_instance
        procedure :: set_instance => vector_int_set_instance
        procedure :: associated => vector_int_associated
        ! splicer begin class.vector_int.type_bound_procedure_part
        ! splicer end class.vector_int.type_bound_procedure_part
    end type vector_int

    interface operator (.eq.)
        module procedure vector_int_eq
    end interface

    interface operator (.ne.)
        module procedure vector_int_ne
    end interface

    interface

        function c_vector_int_ctor() &
                result(SHT_rv) &
                bind(C, name="TEM_vector_int_ctor")
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data) :: SHT_rv
        end function c_vector_int_ctor

        subroutine c_vector_int_dtor(self) &
                bind(C, name="TEM_vector_int_dtor")
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
        end subroutine c_vector_int_dtor

        subroutine c_vector_int_push_back(self, value) &
                bind(C, name="TEM_vector_int_push_back")
            use iso_c_binding, only : C_INT
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_INT), intent(IN) :: value
        end subroutine c_vector_int_push_back

        ! splicer begin class.vector_int.additional_interfaces
        ! splicer end class.vector_int.additional_interfaces
    end interface

contains

    function vector_int_ctor() &
            result(SHT_rv)
        type(vector_int) :: SHT_rv
        ! splicer begin class.vector_int.method.ctor
        SHT_rv%cxxmem = c_vector_int_ctor()
        ! splicer end class.vector_int.method.ctor
    end function vector_int_ctor

    subroutine vector_int_dtor(obj)
        class(vector_int) :: obj
        ! splicer begin class.vector_int.method.dtor
        call c_vector_int_dtor(obj%cxxmem)
        ! splicer end class.vector_int.method.dtor
    end subroutine vector_int_dtor

    subroutine vector_int_push_back(obj, value)
        use iso_c_binding, only : C_INT
        class(vector_int) :: obj
        integer(C_INT), intent(IN) :: value
        ! splicer begin class.vector_int.method.push_back
        call c_vector_int_push_back(obj%cxxmem, value)
        ! splicer end class.vector_int.method.push_back
    end subroutine vector_int_push_back

    ! Return pointer to C++ memory.
    function vector_int_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(vector_int), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function vector_int_get_instance

    subroutine vector_int_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(vector_int), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine vector_int_set_instance

    function vector_int_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(vector_int), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function vector_int_associated

    ! splicer begin class.vector_int.additional_functions
    ! splicer end class.vector_int.additional_functions

    function vector_int_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(vector_int), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vector_int_eq

    function vector_int_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(vector_int), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vector_int_ne

end module vector_int_mod
