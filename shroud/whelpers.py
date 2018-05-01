# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-738041.
# All rights reserved.
#  
# This file is part of Shroud.  For details, see
# https://github.com/LLNL/shroud. Please also read shroud/LICENSE.
#  
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the disclaimer below.
# 
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the disclaimer (as noted below)
#   in the documentation and/or other materials provided with the
#   distribution.
# 
# * Neither the name of the LLNS/LLNL nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL LAWRENCE
# LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
########################################################################
"""
Helper functions for C and Fortran wrappers.
"""

from . import util

wformat = util.wformat

def find_all_helpers(mode, helpers, check=None):
    """Find all helper functions recursively.
    A helper function is required by some argument/result conversions.

    Return helper dictionary used.
    """
    if mode =='c':
        helpdict = CHelpers
    elif mode == 'f':
        helpdict = FHelpers
    else:
        raise RuntimeError("Unexpected mode in find_all_helpers")

    if check is None:
        # do all top level helpers
        # Copy initial keys since helpers may change
        keys = list(helpers.keys())
        for check in keys:
            for name in helpdict[check].get('f_helper', []):
                find_all_helpers(mode, helpers, name)
    else:
        if check not in helpers:
            helpers[check] = True
            for name in helpdict[check].get('f_helper', []):
                find_all_helpers(mode, helpers, name)

    return helpdict

def XXXwrite_helper_files(self, directory):
    """This library file is no longer needed.

    Should be writtento config.c_fortran_dir
    """
    output = [FccHeaders]
    self.write_output_file('shroudrt.hpp',
                           directory, output)

    output = [FccCSource]
    self.write_output_file('shroudrt.cpp',
                           directory, output)

FccHeaders = """
#ifndef SHROUDRT_HPP_
#define SHROUDRT_HPP_

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* SHROUDRT_HPP_ */
"""

#
# C routines used by wrappers
#
# shroud_c_loc must be compiled by C but called by Fortran.
#

FccCSource = """
/* *INDENT-OFF* */
#ifdef __cplusplus
extern "C" {
#else
#endif
/* *INDENT-ON* */

// insert code here

/* *INDENT-OFF* */
#ifdef __cplusplus
}  // extern \"C\"
#endif
/* *INDENT-ON* */"""

#
# C helper functions which may be added to a implementation file.
#
# c_helpers = Dictionary of helpers needed by this helper
# c_header    = Blank delimited list of header files to #include
#               in implementation file when wrapping a C library.
# cxx_header  = Blank delimited list of header files to #include.
#               in implementation file when wrapping a C++ library.
# c_source    = language=c source.
# cxx_source  = language=c++ source.
# h_header    = Blank delimited list of headers to #include in
#               c wrapper header.
# h_source    = code for include file. Must be compatible with language=c.
# source      = Code inserted before any wrappers.
#               The functions should be file static.
#               Used if c_source or cxx_source is not defined.

num_union_helpers = 0
def add_union_helper(cxx, c, num=0):
    """A union helper is used to convert between a struct in C and C++.
    The structs are identical but the names are different. For example,
    the C++ version is in a namespace.

    Make the C++ struct first in the union to make it possible to assign
      name var = { cpp_func() };
    Assigns to var.cxx.

    The struct are named sequentially to generate unique names.
    """
    global num_union_helpers
    name = 'SH_union_{}_t'.format(num_union_helpers)
    num_union_helpers += 1
    if name in CHelpers:
        raise RuntimeError("{} already exists in CHelpers".format(name))
    helper = dict(cxx_source="""
typedef union {{
  {cxx} cxx;
  {c} c;
}} {name};
""".format(name=name, cxx=cxx, c=c))
    CHelpers[name] = helper
    return name

def add_external_helpers(fmt):
    """Create helper which have generated names.
    Since the names are external, mangle with C_prefix to avoid
    confict with other shroud wrapped libraries.
    """

    # Only used with std::string and thus C++
    name = 'copy_string'
    CHelpers[name] = dict(
        cxx_header='<string>',
# XXX - mangle name
        source=wformat("""
// Called by Fortran to deal with allocatable character
void {C_prefix}ShroudStringCopyAndFree(void *cptr, char *str) {{+
std::string * cxxstr = static_cast<std::string *>(cptr);

strncpy(str, cxxstr->data(), cxxstr->size());
// free the string?
-}}
""", fmt)
    )

    # Deal with allocatable character
    FHelpers[name] = dict(
        interface=wformat("""
interface
   subroutine SHROUD_string_copy_and_free(cptr, str) &
     bind(c,name="{C_prefix}ShroudStringCopyAndFree")
     use, intrinsic :: iso_c_binding, only : C_PTR, C_CHAR
     type(C_PTR), value, intent(in) :: cptr
     character(kind=C_CHAR) :: str(*)
   end subroutine SHROUD_string_copy_and_free
end interface""", fmt)
        )
    ##########

def add_capsule_helper(fmt):
    """Share info with C++ to allow Fortran to release memory.

    Used with shadow classes and std::vector.
    """
    name = 'capsule_data_helper'
    if name not in FHelpers:
        helper = dict(
            derived_type=wformat("""
type, bind(C) :: {F_capsule_data_type}+
type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
integer(C_INT) :: idtor = 0       ! index of destructor
-end type {F_capsule_data_type}""", fmt),
            modules = dict(
                iso_c_binding=['C_NULL_PTR', 'C_PTR', 'C_INT' ],
            ),
        )
        FHelpers[name] = helper


    if name not in CHelpers:
        helper = dict(
            h_source=wformat("""
struct s_{C_capsule_data_type} {{
  void *addr;     /* address of C++ memory */
  int idtor;      /* index of destructor */
}};
typedef struct s_{C_capsule_data_type} {C_capsule_data_type};""", fmt),
        )
        CHelpers[name] = helper

    ########################################
    name = 'capsule_helper'
    if name not in FHelpers:
# XXX split helper into to parts, one for each derived type
        helper = dict(
            dependent_helpers=[ 'capsule_data_helper' ],
            derived_type=wformat("""
type {F_capsule_type}+
private
type({F_capsule_data_type}) :: mem
-contains
+final :: {F_capsule_final_function}
-end type {F_capsule_type}""", fmt),
# cannot be declared with both PRIVATE and BIND(C) attributes
            source = wformat("""
subroutine {F_capsule_final_function}(cap)+
type({F_capsule_type}), intent(INOUT) :: cap
interface+
subroutine array_destructor(mem)\tbind(C, name="{C_memory_dtor_function}")+
import {F_capsule_data_type}
implicit none
type({F_capsule_data_type}), intent(INOUT) :: mem      
-end subroutine array_destructor
-end interface
call array_destructor(cap%mem)
-end subroutine {F_capsule_final_function}
            """, fmt),
        )
        FHelpers[name] = helper

def add_vector_copy_helper(fmt):
    """Create function to copy contents of a vector.
    """
    name = wformat('vector_copy_{cxx_T}', fmt)
    if name not in CHelpers:
        helper = dict(
            cxx_source=wformat("""
void {C_prefix}SHROUD_vector_copy_{cxx_T}({C_capsule_data_type} *cap, \t{cxx_T} *c_var, \tsize_t c_var_size)
{{+
std::vector<{cxx_T}> *cxx_var = \treinterpret_cast<std::vector<{cxx_T}> *>\t(cap->addr);
std::vector<{cxx_T}>::size_type+
i = 0,
n = c_var_size;
-n = std::min(cxx_var->size(), n);
for(; i < n; ++i) {{+
c_var[i] = (*cxx_var)[i];
-}}
-}}""", fmt))
        CHelpers[name] = helper
    if name not in FHelpers:
        helper = dict(
# XXX when f_kind == C_SIZE_T
            interface=wformat("""
interface+
subroutine SHROUD_vector_copy_{cxx_T}(cap, c_var, c_var_size) &+
bind(C, name="{C_prefix}SHROUD_vector_copy_{cxx_T}")
use iso_c_binding, only : {f_kind}, C_SIZE_T
import {C_capsule_data_type}
type({C_capsule_data_type}) :: cap
integer({f_kind}) :: c_var(*)
integer(C_SIZE_T), value :: c_var_size
-end subroutine SHROUD_vector_copy_{cxx_T}
-end interface
""", fmt),
        )
        FHelpers[name] = helper
    return name

CHelpers = dict(
    ShroudStrCopy=dict(
        c_header='<string.h>',
        cxx_header='<cstring>',
        c_source="""
// Copy s into a, blank fill to la characters
// Truncate if a is too short.
static void ShroudStrCopy(char *a, int la, const char *s)
{
   int ls,nm;
   ls = strlen(s);
   nm = ls < la ? ls : la;
   memcpy(a,s,nm);
   if(la > nm) memset(a+nm,' ',la-nm);
}""",
        cxx_source="""
// Copy s into a, blank fill to la characters
// Truncate if a is too short.
static void ShroudStrCopy(char *a, int la, const char *s)
{
   int ls,nm;
   ls = std::strlen(s);
   nm = ls < la ? ls : la;
   std::memcpy(a,s,nm);
   if(la > nm) std::memset(a+nm,' ',la-nm);
}"""
        ),
    ShroudLenTrim=dict(
        source="""
// Returns the length of character string a with length ls,
// ignoring any trailing blanks.
int ShroudLenTrim(const char *s, int ls) {
    int i;

    for (i = ls - 1; i >= 0; i--) {
        if (s[i] != ' ') {
            break;
        }
    }

    return i + 1;
}
"""
    ),

    vector_context=dict(
        h_header='<stddef.h>',
        h_source="""
struct s_SHROUD_vector_context {
  void *addr;     /* address of data in std::vector */
  size_t size;    /* size of data in std::vector */
};
typedef struct s_SHROUD_vector_context SHROUD_vector_context;
"""
    ),

    ) # end CHelpers

#
# Fortran helper functions which may be added to a module.
#
# dependent_helpers = list of helpers names needed by this helper
#                     They will be added to the output before current helper.
# private   = names for PRIVATE statement 
# interface = code for INTERFACE
# source    = code for CONTAINS

FHelpers = dict(
    fstr=dict(
        f_helper=dict(fstr_ptr=True, fstr_arr=True),
        dependent_helpers=[ 'fstr_ptr', 'fstr_arr' ],
        private=['fstr'],
        interface="""
interface fstr
  module procedure fstr_ptr, fstr_arr
end interface""",
        ),

    fstr_ptr=dict(
        f_helper=dict(strlen_ptr=True),
        dependent_helpers=[ 'strlen_ptr' ],
        private=['fstr_ptr'],
        source="""
! Convert a null-terminated C "char *" pointer to a Fortran string.
function fstr_ptr(s) result(fs)
  use, intrinsic :: iso_c_binding, only: c_char, c_ptr, c_f_pointer
  type(c_ptr), intent(in) :: s
  character(kind=c_char, len=strlen_ptr(s)) :: fs
  character(kind=c_char), pointer :: cptr(:)
  integer :: i
  call c_f_pointer(s, cptr, [len(fs)])
  do i=1, len(fs)
     fs(i:i) = cptr(i)
  enddo
end function fstr_ptr"""
        ),

    fstr_arr=dict(
        f_helper=dict(strlen_arr=True),
        dependent_helpers=[ 'strlen_arr' ],
        private=['fstr_arr'],
        source="""
! Convert a null-terminated array of characters to a Fortran string.
function fstr_arr(s) result(fs)
  use, intrinsic :: iso_c_binding, only : c_char, c_null_char
  character(kind=c_char, len=1), intent(in) :: s(*)
  character(kind=c_char, len=strlen_arr(s)) :: fs
  integer :: i
  do i = 1, len(fs)
     fs(i:i) = s(i)
  enddo
end function fstr_arr"""
        ),

    strlen_arr=dict(
        private=['strlen_arr'],
        source="""
! Count the characters in a null-terminated array.
pure function strlen_arr(s)
  use, intrinsic :: iso_c_binding, only : c_char, c_null_char
  character(kind=c_char, len=1), intent(in) :: s(*)
  integer :: i, strlen_arr
  i=1
  do
     if (s(i) == c_null_char) exit
     i = i+1
  enddo
  strlen_arr = i-1
end function strlen_arr"""
    ),

    strlen_ptr=dict(
        private=['strlen_ptr'],
        interface="""
interface
   pure function strlen_ptr(s) result(result) bind(c,name="strlen")
     use, intrinsic :: iso_c_binding
     integer(c_int) :: result
     type(c_ptr), value, intent(in) :: s
   end function strlen_ptr
end interface"""
        ),

    # Create a derived type used to communicate with C wrapper.
    # Should never be exposed to user.
    vector_context=dict(
        derived_type="""
type, bind(C) :: SHROUD_vector_context
  type(C_PTR) :: addr     ! address of data in std::vector
  integer(C_SIZE_T) :: size  ! size of data in std::vector
end type SHROUD_vector_context
""",
        modules = dict(
            iso_c_binding=['C_PTR', 'C_SIZE_T' ],
        )
    ),

    ) # end FHelpers


# From fstr_mod.f
#  ! Convert a fortran string in 's' to a null-terminated array of characters.
#  pure function cstr(s)
#    use, intrinsic :: iso_c_binding, only : c_char, c_null_char
#    character(len=*), intent(in) :: s
#    character(kind=c_char, len=1) :: cstr(len_trim(s)+1)
#    integer :: i
#    if (len_trim(s) > 0) cstr = [ (s(i:i), i=1,len_trim(s)) ]
#    cstr(len_trim(s)+1) = c_null_char
#  end function cstr


cmake = """
# Setup Shroud
# This file defines:
#  SHROUD_FOUND - If Shroud was found

if(NOT SHROUD_EXECUTABLE)
    MESSAGE(FATAL_ERROR "Could not find Shroud. Shroud requires explicit SHROUD_EXECUTABLE.")
endif()

message(STATUS "Found SHROUD: ${SHROUD_EXECUTABLE}")

add_custom_target(generate)
set(SHROUD_FOUND TRUE)

#
# Setup targets to generate code.
#
# Each package can create their own ${PROJECT}_generate target
#  add_dependencies(generate  ${PROJECT}_generate)

##------------------------------------------------------------------------------
## add_shroud( YAML_INPUT_FILE file
##             DEPENDS_SOURCE file1 ... filen
##             DEPENDS_BINARY file1 ... filen
##             C_FORTRAN_OUTPUT_DIR dir
##             PYTHON_OUTPUT_DIR dir
##             LUA_OUTPUT_DIR dir
##             YAML_OUTPUT_DIR dir
##             CFILES file
##             FFILES file
## )
##
##  YAML_INPUT_FILE - yaml input file to shroud. Required.
##  DEPENDS_SOURCE  - splicer files in the source directory
##  DEPENDS_BINARY  - splicer files in the binary directory
##  C_FORTRAN_OUTPUT_DIR - directory for C and Fortran wrapper output files.
##  PYTHON_OUTPUT_DIR - directory for Python wrapper output files.
##  LUA_OUTPUT_DIR  - directory for Lua wrapper output files.
##  YAML_OUTPUT_DIR - directory for YAML output files.
##                    Defaults to CMAKE_CURRENT_SOURCE_DIR
##  CFILES          - Output file with list of generated C/C++ files
##  FFILES          - Output file with list of generated Fortran files
##
## Add a target generate_${basename} where basename is generated from
## YAML_INPUT_FILE.  It is then added as a dependency to the generate target.
##
##------------------------------------------------------------------------------

macro(add_shroud)

    # Decide where the output files should be written.
    # For now all files are written into the source directory.
    # This allows them to be source controlled and does not require a library user
    # to generate them.  All they have to do is compile them.
    #set(SHROUD_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    set(SHROUD_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

    set(options)
    set(singleValueArgs
        YAML_INPUT_FILE
        C_FORTRAN_OUTPUT_DIR
        PYTHON_OUTPUT_DIR
        LUA_OUTPUT_DIR
        YAML_OUTPUT_DIR
        CFILES
        FFILES
    )
    set(multiValueArgs DEPENDS_SOURCE DEPENDS_BINARY )

    ## parse the arguments to the macro
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    # make sure YAML_INPUT_FILE is defined
    if(NOT arg_YAML_INPUT_FILE)
      message(FATAL_ERROR "add_shroud macro must define YAML_INPUT_FILE")
    endif()
    get_filename_component(_basename ${arg_YAML_INPUT_FILE} NAME_WE)

    if(arg_C_FORTRAN_OUTPUT_DIR)
      set(SHROUD_C_FORTRAN_OUTPUT_DIR --outdir-c-fortran ${arg_C_FORTRAN_OUTPUT_DIR})
    endif()

    if(arg_PYTHON_OUTPUT_DIR)
      set(SHROUD_PYTHON_OUTPUT_DIR --outdir-python ${arg_PYTHON_OUTPUT_DIR})
    endif()

    if(arg_LUA_OUTPUT_DIR)
      set(SHROUD_LUA_OUTPUT_DIR --outdir-lua ${arg_LUA_OUTPUT_DIR})
    endif()

    if(arg_YAML_OUTPUT_DIR)
      set(SHROUD_YAML_OUTPUT_DIR --outdir-yaml ${arg_YAML_OUTPUT_DIR})
    else()
      set(SHROUD_YAML_OUTPUT_DIR --outdir-yaml ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    if(arg_CFILES)
      set(SHROUD_CFILES ${arg_CFILES})
    else()
      set(SHROUD_CFILES ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.cfiles)
    endif()

    if(arg_FFILES)
      set(SHROUD_FFILES ${arg_FFILES})
    else()
      set(SHROUD_FFILES ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.ffiles)
    endif()

    # convert DEPENDS to full paths
    set(shroud_depends)
    foreach (_file ${arg_DEPENDS_SOURCE})
        list(APPEND shroud_depends "${CMAKE_CURRENT_SOURCE_DIR}/${_file}")
    endforeach ()
    foreach (_file ${arg_DEPENDS_BINARY})
        list(APPEND shroud_depends "${CMAKE_CURRENT_BINARY_DIR}/${_file}")
    endforeach ()

    set(_timestamp  ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.time)

    set(_cmd
        ${SHROUD_EXECUTABLE}
        --logdir ${CMAKE_CURRENT_BINARY_DIR}
        ${SHROUD_C_FORTRAN_OUTPUT_DIR}
        ${SHROUD_PYTHON_OUTPUT_DIR}
        ${SHROUD_LUA_OUTPUT_DIR}
        ${SHROUD_YAML_OUTPUT_DIR}
        # path controls where to search for splicer files listed in YAML_INPUT_FILE
        --path ${CMAKE_CURRENT_BINARY_DIR}
        --path ${CMAKE_CURRENT_SOURCE_DIR}
        --cfiles ${SHROUD_CFILES}
        --ffiles ${SHROUD_FFILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/${arg_YAML_INPUT_FILE}
    )

    add_custom_command(
        OUTPUT  ${_timestamp}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${arg_YAML_INPUT_FILE} ${shroud_depends}
        COMMAND ${_cmd}
        COMMAND touch ${_timestamp}
        COMMENT "Running shroud ${arg_YAML_INPUT_FILE}"
        WORKING_DIRECTORY ${SHROUD_OUTPUT_DIR}
    )

    # Create target to process this Shroud file
    add_custom_target(generate_${_basename}    DEPENDS ${_timestamp})

    add_dependencies(generate generate_${_basename})
endmacro(add_shroud)
"""
