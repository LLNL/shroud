! wrapftemplates_std.f
! This is generated code, do not edit
!>
!! \file wrapftemplates_std.f
!! \brief Shroud generated wrapper for std namespace
!<
! splicer begin namespace.std.file_top
! splicer end namespace.std.file_top
module templates_std_mod
    use iso_c_binding, only : C_INT, C_NULL_PTR, C_PTR
    ! splicer begin namespace.std.module_use
    ! splicer end namespace.std.module_use
    implicit none

    ! splicer begin namespace.std.module_top
    ! splicer end namespace.std.module_top

    type, bind(C) :: SHROUD_vector_int_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_vector_int_capsule

    type vector_int
        type(SHROUD_vector_int_capsule) :: cxxmem
        ! splicer begin namespace.std.class.vector_int.component_part
        ! splicer end namespace.std.class.vector_int.component_part
    contains
        procedure :: dtor => vector_int_dtor
        procedure :: push_back => vector_int_push_back
        procedure :: at => vector_int_at
        procedure :: get_instance => vector_int_get_instance
        procedure :: set_instance => vector_int_set_instance
        procedure :: associated => vector_int_associated
        ! splicer begin namespace.std.class.vector_int.type_bound_procedure_part
        ! splicer end namespace.std.class.vector_int.type_bound_procedure_part
    end type vector_int

    type, bind(C) :: SHROUD_vector_double_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_vector_double_capsule

    type vector_double
        type(SHROUD_vector_double_capsule) :: cxxmem
        ! splicer begin namespace.std.class.vector_double.component_part
        ! splicer end namespace.std.class.vector_double.component_part
    contains
        procedure :: dtor => vector_double_dtor
        procedure :: push_back => vector_double_push_back
        procedure :: at => vector_double_at
        procedure :: get_instance => vector_double_get_instance
        procedure :: set_instance => vector_double_set_instance
        procedure :: associated => vector_double_associated
        ! splicer begin namespace.std.class.vector_double.type_bound_procedure_part
        ! splicer end namespace.std.class.vector_double.type_bound_procedure_part
    end type vector_double

    interface operator (.eq.)
        module procedure vector_int_eq
        module procedure vector_double_eq
    end interface

    interface operator (.ne.)
        module procedure vector_int_ne
        module procedure vector_double_ne
    end interface

    interface

        function c_vector_int_ctor(SHT_crv) &
                result(SHT_rv) &
                bind(C, name="TEM_vector_int_ctor")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_vector_int_capsule
            implicit none
            type(SHROUD_vector_int_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_vector_int_ctor

        subroutine c_vector_int_dtor(self) &
                bind(C, name="TEM_vector_int_dtor")
            import :: SHROUD_vector_int_capsule
            implicit none
            type(SHROUD_vector_int_capsule), intent(IN) :: self
        end subroutine c_vector_int_dtor

        subroutine c_vector_int_push_back(self, value) &
                bind(C, name="TEM_vector_int_push_back")
            use iso_c_binding, only : C_INT
            import :: SHROUD_vector_int_capsule
            implicit none
            type(SHROUD_vector_int_capsule), intent(IN) :: self
            integer(C_INT), intent(IN) :: value
        end subroutine c_vector_int_push_back

        function c_vector_int_at(self, n) &
                result(SHT_rv) &
                bind(C, name="TEM_vector_int_at")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            import :: SHROUD_vector_int_capsule
            implicit none
            type(SHROUD_vector_int_capsule), intent(IN) :: self
            integer(C_SIZE_T), value, intent(IN) :: n
            type(C_PTR) SHT_rv
        end function c_vector_int_at

        ! splicer begin namespace.std.class.vector_int.additional_interfaces
        ! splicer end namespace.std.class.vector_int.additional_interfaces

        function c_vector_double_ctor(SHT_crv) &
                result(SHT_rv) &
                bind(C, name="TEM_vector_double_ctor")
            use iso_c_binding, only : C_PTR
            import :: SHROUD_vector_double_capsule
            implicit none
            type(SHROUD_vector_double_capsule), intent(OUT) :: SHT_crv
            type(C_PTR) SHT_rv
        end function c_vector_double_ctor

        subroutine c_vector_double_dtor(self) &
                bind(C, name="TEM_vector_double_dtor")
            import :: SHROUD_vector_double_capsule
            implicit none
            type(SHROUD_vector_double_capsule), intent(IN) :: self
        end subroutine c_vector_double_dtor

        subroutine c_vector_double_push_back(self, value) &
                bind(C, name="TEM_vector_double_push_back")
            use iso_c_binding, only : C_DOUBLE
            import :: SHROUD_vector_double_capsule
            implicit none
            type(SHROUD_vector_double_capsule), intent(IN) :: self
            real(C_DOUBLE), intent(IN) :: value
        end subroutine c_vector_double_push_back

        function c_vector_double_at(self, n) &
                result(SHT_rv) &
                bind(C, name="TEM_vector_double_at")
            use iso_c_binding, only : C_PTR, C_SIZE_T
            import :: SHROUD_vector_double_capsule
            implicit none
            type(SHROUD_vector_double_capsule), intent(IN) :: self
            integer(C_SIZE_T), value, intent(IN) :: n
            type(C_PTR) SHT_rv
        end function c_vector_double_at

        ! splicer begin namespace.std.class.vector_double.additional_interfaces
        ! splicer end namespace.std.class.vector_double.additional_interfaces

        ! splicer begin namespace.std.additional_interfaces
        ! splicer end namespace.std.additional_interfaces
    end interface

contains

    function vector_int_ctor() &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        type(vector_int) :: SHT_rv
        ! splicer begin namespace.std.class.vector_int.method.ctor
        type(C_PTR) :: SHT_prv
        SHT_prv = c_vector_int_ctor(SHT_rv%cxxmem)
        ! splicer end namespace.std.class.vector_int.method.ctor
    end function vector_int_ctor

    subroutine vector_int_dtor(obj)
        class(vector_int) :: obj
        ! splicer begin namespace.std.class.vector_int.method.dtor
        call c_vector_int_dtor(obj%cxxmem)
        ! splicer end namespace.std.class.vector_int.method.dtor
    end subroutine vector_int_dtor

    subroutine vector_int_push_back(obj, value)
        use iso_c_binding, only : C_INT
        class(vector_int) :: obj
        integer(C_INT), intent(IN) :: value
        ! splicer begin namespace.std.class.vector_int.method.push_back
        call c_vector_int_push_back(obj%cxxmem, value)
        ! splicer end namespace.std.class.vector_int.method.push_back
    end subroutine vector_int_push_back

    function vector_int_at(obj, n) &
            result(SHT_rv)
        use iso_c_binding, only : C_INT, C_PTR, C_SIZE_T, c_f_pointer
        class(vector_int) :: obj
        integer(C_SIZE_T), value, intent(IN) :: n
        integer(C_INT), pointer :: SHT_rv
        ! splicer begin namespace.std.class.vector_int.method.at
        type(C_PTR) :: SHT_ptr
        SHT_ptr = c_vector_int_at(obj%cxxmem, n)
        call c_f_pointer(SHT_ptr, SHT_rv)
        ! splicer end namespace.std.class.vector_int.method.at
    end function vector_int_at

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

    ! splicer begin namespace.std.class.vector_int.additional_functions
    ! splicer end namespace.std.class.vector_int.additional_functions

    function vector_double_ctor() &
            result(SHT_rv)
        use iso_c_binding, only : C_PTR
        type(vector_double) :: SHT_rv
        ! splicer begin namespace.std.class.vector_double.method.ctor
        type(C_PTR) :: SHT_prv
        SHT_prv = c_vector_double_ctor(SHT_rv%cxxmem)
        ! splicer end namespace.std.class.vector_double.method.ctor
    end function vector_double_ctor

    subroutine vector_double_dtor(obj)
        class(vector_double) :: obj
        ! splicer begin namespace.std.class.vector_double.method.dtor
        call c_vector_double_dtor(obj%cxxmem)
        ! splicer end namespace.std.class.vector_double.method.dtor
    end subroutine vector_double_dtor

    subroutine vector_double_push_back(obj, value)
        use iso_c_binding, only : C_DOUBLE
        class(vector_double) :: obj
        real(C_DOUBLE), intent(IN) :: value
        ! splicer begin namespace.std.class.vector_double.method.push_back
        call c_vector_double_push_back(obj%cxxmem, value)
        ! splicer end namespace.std.class.vector_double.method.push_back
    end subroutine vector_double_push_back

    function vector_double_at(obj, n) &
            result(SHT_rv)
        use iso_c_binding, only : C_DOUBLE, C_PTR, C_SIZE_T, c_f_pointer
        class(vector_double) :: obj
        integer(C_SIZE_T), value, intent(IN) :: n
        real(C_DOUBLE), pointer :: SHT_rv
        ! splicer begin namespace.std.class.vector_double.method.at
        type(C_PTR) :: SHT_ptr
        SHT_ptr = c_vector_double_at(obj%cxxmem, n)
        call c_f_pointer(SHT_ptr, SHT_rv)
        ! splicer end namespace.std.class.vector_double.method.at
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

    ! splicer begin namespace.std.class.vector_double.additional_functions
    ! splicer end namespace.std.class.vector_double.additional_functions

    ! splicer begin namespace.std.additional_functions
    ! splicer end namespace.std.additional_functions

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

end module templates_std_mod
