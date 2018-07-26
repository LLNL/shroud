! wrapfvector_double.f
! This is generated code, do not edit
!>
!! \file wrapfvector_double.f
!! \brief Shroud generated wrapper for vector class
!<
! splicer begin file_top
! splicer end file_top
module vector_double_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin class.vector_double.module_use
    ! splicer end class.vector_double.module_use
    implicit none


    type, bind(C) :: SHROUD_capsule_data
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_capsule_data

    ! splicer begin class.vector_double.module_top
    ! splicer end class.vector_double.module_top

    type vector_double
        type(SHROUD_capsule_data) :: cxxmem
        ! splicer begin class.vector_double.component_part
        ! splicer end class.vector_double.component_part
    contains
        procedure :: dtor => vector_double_dtor
        procedure :: push_back => vector_double_push_back
        procedure :: at => vector_double_at
        procedure :: get_instance => vector_double_get_instance
        procedure :: set_instance => vector_double_set_instance
        procedure :: associated => vector_double_associated
        ! splicer begin class.vector_double.type_bound_procedure_part
        ! splicer end class.vector_double.type_bound_procedure_part
    end type vector_double

    interface operator (.eq.)
        module procedure vector_double_eq
    end interface

    interface operator (.ne.)
        module procedure vector_double_ne
    end interface

    interface

        function c_vector_double_ctor() &
                result(SHT_rv) &
                bind(C, name="TEM_vector_double_ctor")
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data) :: SHT_rv
        end function c_vector_double_ctor

        subroutine c_vector_double_dtor(self) &
                bind(C, name="TEM_vector_double_dtor")
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
        end subroutine c_vector_double_dtor

        subroutine c_vector_double_push_back(self, value) &
                bind(C, name="TEM_vector_double_push_back")
            use iso_c_binding, only : C_DOUBLE
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            real(C_DOUBLE), intent(IN) :: value
        end subroutine c_vector_double_push_back

        function c_vector_double_at(self, n) &
                result(SHT_rv) &
                bind(C, name="TEM_vector_double_at")
            use iso_c_binding, only : C_DOUBLE, C_PTR, C_SIZE_T
            import :: SHROUD_capsule_data
            implicit none
            type(SHROUD_capsule_data), intent(IN) :: self
            integer(C_SIZE_T), value, intent(IN) :: n
            type(C_PTR) SHT_rv
        end function c_vector_double_at

        ! splicer begin class.vector_double.additional_interfaces
        ! splicer end class.vector_double.additional_interfaces
    end interface

contains

    function vector_double_ctor() &
            result(SHT_rv)
        type(vector_double) :: SHT_rv
        ! splicer begin class.vector_double.method.ctor
        SHT_rv%cxxmem = c_vector_double_ctor()
        ! splicer end class.vector_double.method.ctor
    end function vector_double_ctor

    subroutine vector_double_dtor(obj)
        class(vector_double) :: obj
        ! splicer begin class.vector_double.method.dtor
        call c_vector_double_dtor(obj%cxxmem)
        ! splicer end class.vector_double.method.dtor
    end subroutine vector_double_dtor

    subroutine vector_double_push_back(obj, value)
        use iso_c_binding, only : C_DOUBLE
        class(vector_double) :: obj
        real(C_DOUBLE), intent(IN) :: value
        ! splicer begin class.vector_double.method.push_back
        call c_vector_double_push_back(obj%cxxmem, value)
        ! splicer end class.vector_double.method.push_back
    end subroutine vector_double_push_back

    function vector_double_at(obj, n) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE, C_PTR, C_SIZE_T, c_f_pointer
        class(vector_double) :: obj
        integer(C_SIZE_T), value, intent(IN) :: n
        real(C_DOUBLE), pointer :: SHT_rv
        type(C_PTR) :: SHT_ptr
        ! splicer begin class.vector_double.method.at
        SHT_ptr = c_vector_double_at(obj%cxxmem, n)
        call c_f_pointer(SHT_ptr, SHT_rv)
        ! splicer end class.vector_double.method.at
    end function vector_double_at

    ! Return pointer to C++ memory.
    function vector_double_get_instance(obj) result (cxxptr)
        use iso_c_binding, only: C_PTR
        class(vector_double), intent(IN) :: obj
        type(C_PTR) :: cxxptr
        cxxptr = obj%cxxmem%addr
    end function vector_double_get_instance

    subroutine vector_double_set_instance(obj, cxxmem)
        use iso_c_binding, only: C_PTR
        class(vector_double), intent(INOUT) :: obj
        type(C_PTR), intent(IN) :: cxxmem
        obj%cxxmem%addr = cxxmem
        obj%cxxmem%idtor = 0
    end subroutine vector_double_set_instance

    function vector_double_associated(obj) result (rv)
        use iso_c_binding, only: c_associated
        class(vector_double), intent(IN) :: obj
        logical rv
        rv = c_associated(obj%cxxmem%addr)
    end function vector_double_associated

    ! splicer begin class.vector_double.additional_functions
    ! splicer end class.vector_double.additional_functions

    function vector_double_eq(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(vector_double), intent(IN) ::a,b
        logical :: rv
        if (c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vector_double_eq

    function vector_double_ne(a,b) result (rv)
        use iso_c_binding, only: c_associated
        type(vector_double), intent(IN) ::a,b
        logical :: rv
        if (.not. c_associated(a%cxxmem%addr, b%cxxmem%addr)) then
            rv = .true.
        else
            rv = .false.
        endif
    end function vector_double_ne

end module vector_double_mod
