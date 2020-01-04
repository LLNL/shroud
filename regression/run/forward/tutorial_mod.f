!
! Another package which has been 'wrapped' by shroud
!
module tutorial_mod

    use iso_c_binding, only : C_INT, C_PTR, C_NULL_PTR

    type, bind(C) :: SHROUD_class1_capsule
        type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
        integer(C_INT) :: idtor = 0       ! index of destructor
    end type SHROUD_class1_capsule

    type class1
        type(SHROUD_class1_capsule) :: cxxmem
    end type class1
  
end module tutorial_mod
