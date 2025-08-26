! Copyright Shroud Project Developers. See LICENSE file for details.
!
! From the struct-cxx test reference

module struct_mod
  use iso_c_binding, only : C_DOUBLE, C_INT

  type, bind(C) :: cstruct1
     integer(C_INT) :: ifield
     real(C_DOUBLE) :: dfield
  end type cstruct1

end module struct_mod
