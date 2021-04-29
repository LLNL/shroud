#
# Check Fortran compiler features required for Shroud
#

cmake_minimum_required(VERSION 3.1)
project(tester LANGUAGES Fortran C)

# CheckFortranSourceCompilers  3.1
# CheckFortranSourceRuns       3.14

include(CheckCSourceCompiles)
include(CheckFortranSourceCompiles)

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
