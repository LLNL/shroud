#
# Check Fortran compiler features required for Shroud
#

cmake_minimum_required(VERSION 3.14)
project(tester LANGUAGES Fortran C)

# CheckFortranSourceCompilers  3.1
# CheckFortranSourceRuns       3.14

include(CheckCSourceCompiles)
include(CheckFortranSourceCompiles)
include(CheckFortranSourceRuns)

########################################
check_fortran_source_compiles("
  program tester
    interface
      subroutine ctester(arg) bind(C)
        type(*) arg
      end subroutine ctester
    end interface
  end program
"
  HAVE_ASSUMED_TYPE
  SRC_EXT f90
)

########################################
# Test for feature used by strings.yaml
# - decl: const char * getCharPtr5() +deref(pointer)
# Function declared as CHARACTER(len=:), POINTER

check_fortran_source_runs("
program tester
  use iso_c_binding
  implicit none
  character(30) :: global
  character(len=:), pointer :: result

  global = \"test string\"
  result => get_pointer(global)
  if (len(result) .ne. len_trim(global)) stop 1
  if (result .ne. global) stop 1
  print *, result

contains
  function get_pointer(str) result(rv)
    character(*), intent(IN), target :: str
    character(len=:), pointer :: rv
    type(C_PTR) cptr
    cptr = C_LOC(str)
    call helper(cptr, len_trim(str), rv)
! gfortran crashes with this line.
!    call helper(C_LOC(str), len_trim(str), rv)
  end function get_pointer

  subroutine helper(cptr, nchar, var)
      type(C_PTR), intent(IN) :: cptr
      integer(C_INT), intent(IN) :: nchar
      character(len=:), pointer, intent(OUT) :: var
      character(len=nchar), pointer :: fptr
      call c_f_pointer(cptr, fptr)
      var => fptr
  end subroutine helper
end program tester
"
  HAVE_CHARACTER_POINTER_FUNCTION
)

########################################
check_c_source_compiles("
#include <ISO_Fortran_binding.h>

int main(int argc, char *argv[])
{
    return 0;
}
"
  HAVE_TS29113
)

########################################
configure_file(config.h.in config.h)
