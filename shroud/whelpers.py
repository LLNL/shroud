# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-738041.
#
# All rights reserved.
#
# This file is part of Shroud.
#
# For details about use and distribution, please read LICENSE.
#
########################################################################
"""
Helper functions for C and Fortran wrappers.


 C helper functions which may be added to a implementation file.

 c_header    = Blank delimited list of header files to #include
               in implementation file when wrapping a C library.
 cxx_header  = Blank delimited list of header files to #include.
               in implementation file when wrapping a C++ library.
 c_source    = language=c source.
 cxx_source  = language=c++ source.
 dependent_helpers = list of helpers names needed by this helper
                     They will be added to the output before current helper.

 h_header    = Blank delimited list of headers to #include in
               c wrapper header.
 h_source    = code for include file. Must be compatible with language=c.

 h_shared_include = include files needed by shared header.
 h_shared_code    = code written to C_header_helper file.
                    Useful for struct and typedefs.

 source      = Code inserted before any wrappers.
               The functions should be file static.
               Used if c_source or cxx_source is not defined.


 Fortran helper functions which may be added to a module.

 dependent_helpers = list of helpers names needed by this helper
                     They will be added to the output before current helper.
 private   = names for PRIVATE statement
 interface = code for INTERFACE
 source    = code for CONTAINS



"""

from . import util

wformat = util.wformat


def XXXwrite_helper_files(self, directory):
    """This library file is no longer needed.

    Should be writtento config.c_fortran_dir
    """
    output = [FccHeaders]
    self.write_output_file("shroudrt.hpp", directory, output)

    output = [FccCSource]
    self.write_output_file("shroudrt.cpp", directory, output)


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

num_union_helpers = 0


def add_union_helper(cxx, c, num=0):
    """A union helper is used to convert between a struct in C and C++.
    The structs are identical but the names are different. For example,
    the C++ version is in a namespace.

    Make the C++ struct first in the union to make it possible to assign
      name var = { cpp_func() };
    Assigns to var.cxx.

    The struct are named sequentially to generate unique names.

    Args:
        cxx -
        c -
        num -
    """
    global num_union_helpers
    name = "SH_union_{}_t".format(num_union_helpers)
    num_union_helpers += 1
    if name in CHelpers:
        raise RuntimeError("{} already exists in CHelpers".format(name))
    helper = dict(
        cxx_source="""
typedef union {{
  {cxx} cxx;
  {c} c;
}} {name};
""".format(
            name=name, cxx=cxx, c=c
        )
    )
    CHelpers[name] = helper
    return name


def add_external_helpers(fmt):
    """Create helper which have generated names.
    For example, code uses format entries
    C_prefix, C_memory_dtor_function,
    F_array_type

    Some helpers are written in C, but called by Fortran.
    Since the names are external, mangle with C_prefix to avoid
    confict with other Shroud wrapped libraries.

    Args:
        fmt - format dictionary from the library.
    """

    # Only used with std::string and thus C++
    name = "copy_string"
    CHelpers[name] = dict(
        dependent_helpers=["array_context"],
        cxx_header="<string> <cstddef>",
        # XXX - mangle name
        source=wformat(
            """
// helper function
// Copy the char* or std::string in context into c_var.
// Called by Fortran to deal with allocatable character.
void {C_prefix}ShroudCopyStringAndFree({C_array_type} *data, char *c_var, size_t c_var_len) {{+
const char *cxx_var = data->addr.ccharp;
size_t n = c_var_len;
if (data->len < n) n = data->len;
strncpy(c_var, cxx_var, n);
{C_memory_dtor_function}(&data->cxx); // delete data->cxx.addr
-}}
""",
            fmt,
        ),
    )

    # Deal with allocatable character
    FHelpers[name] = dict(
        dependent_helpers=["array_context"],
        interface=wformat(
            """
interface+
! helper function
! Copy the char* or std::string in context into c_var.
subroutine SHROUD_copy_string_and_free(context, c_var, c_var_size) &
     bind(c,name="{C_prefix}ShroudCopyStringAndFree")+
use, intrinsic :: iso_c_binding, only : C_CHAR, C_SIZE_T
import {F_array_type}
type({F_array_type}), intent(IN) :: context
character(kind=C_CHAR), intent(OUT) :: c_var(*)
integer(C_SIZE_T), value :: c_var_size
-end subroutine SHROUD_copy_string_and_free
-end interface""",
            fmt,
        ),
    )
    ##########


def add_shadow_helper(node):
    """
    Add helper functions for each shadow type.

    Args:
        node -
    """
    cname = node.typemap.c_type

    name = "capsule_{}".format(cname)
    if name not in CHelpers:
        helper = dict(
            h_shared_code="""
struct s_{C_type_name} {{+
void *addr;     /* address of C++ memory */
int idtor;      /* index of destructor */
-}};
typedef struct s_{C_type_name} {C_type_name};""".format(
                C_type_name=cname
            )
        )
        CHelpers[name] = helper
    return name


def add_capsule_helper(fmt):
    """Share info with C++ to allow Fortran to release memory.

    Used with shadow classes and std::vector.

    Args:
        fmt - format dictionary from the library.
    """
    name = "capsule_data_helper"
    if name not in FHelpers:
        helper = dict(
            derived_type=wformat(
                """
type, bind(C) :: {F_capsule_data_type}+
type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
integer(C_INT) :: idtor = 0       ! index of destructor
-end type {F_capsule_data_type}""",
                fmt,
            ),
            modules=dict(iso_c_binding=["C_PTR", "C_INT", "C_NULL_PTR"]),
        )
        FHelpers[name] = helper

    if name not in CHelpers:
        helper = dict(
            h_shared_code=wformat(
                """
struct s_{C_capsule_data_type} {{+
void *addr;     /* address of C++ memory */
int idtor;      /* index of destructor */
-}};
typedef struct s_{C_capsule_data_type} {C_capsule_data_type};""",
                fmt,
            )
        )
        CHelpers[name] = helper

    ########################################
    name = "capsule_helper"
    if name not in FHelpers:
        # XXX split helper into to parts, one for each derived type
        helper = dict(
            dependent_helpers=["capsule_data_helper"],
            derived_type=wformat(
                """
type {F_capsule_type}+
private
type({F_capsule_data_type}) :: mem
-contains
+final :: {F_capsule_final_function}
-end type {F_capsule_type}""",
                fmt,
            ),
            # cannot be declared with both PRIVATE and BIND(C) attributes
            source=wformat(
                """
! finalize a static {F_capsule_data_type}
subroutine {F_capsule_final_function}(cap)+
use iso_c_binding, only : C_BOOL
type({F_capsule_type}), intent(INOUT) :: cap
interface+
subroutine array_destructor(ptr, gc)\tbind(C, name="{C_memory_dtor_function}")+
use iso_c_binding, only : C_BOOL
import {F_capsule_data_type}
implicit none
type({F_capsule_data_type}), intent(INOUT) :: ptr
logical(C_BOOL), value, intent(IN) :: gc
-end subroutine array_destructor
-end interface
call array_destructor(cap%mem, .false._C_BOOL)
-end subroutine {F_capsule_final_function}
            """,
                fmt,
            ),
        )
        FHelpers[name] = helper

    ########################################
    name = "array_context"
    if name not in CHelpers:
        helper = dict(
            h_shared_include="<stddef.h>",
            # Create a union for addr to avoid some casts.
            # And help with debugging since ccharp will display contents.
            h_shared_code=wformat(
                """
struct s_{C_array_type} {{+
{C_capsule_data_type} cxx;      /* address of C++ memory */
union {{+
const void * cvoidp;
const char * ccharp;
-}} addr;
size_t len;     /* bytes-per-item or character len of data in cxx */
size_t size;    /* size of data in cxx */
-}};
typedef struct s_{C_array_type} {C_array_type};""",
                fmt,
            ),
            dependent_helpers=["capsule_data_helper"],
        )
        CHelpers[name] = helper

    if name not in FHelpers:
        # Create a derived type used to communicate with C wrapper.
        # Should never be exposed to user.
        helper = dict(
            derived_type=wformat(
                """
type, bind(C) :: {F_array_type}+
type({F_capsule_data_type}) :: cxx       ! address of C++ memory
type(C_PTR) :: addr = C_NULL_PTR       ! address of data in cxx
integer(C_SIZE_T) :: len = 0_C_SIZE_T  ! bytes-per-item or character len of data in cxx
integer(C_SIZE_T) :: size = 0_C_SIZE_T ! size of data in cxx
-end type {F_array_type}""",
                fmt,
            ),
            modules=dict(iso_c_binding=["C_NULL_PTR", "C_PTR", "C_SIZE_T"]),
            dependent_helpers=["capsule_data_helper"],
        )
        FHelpers[name] = helper


def add_copy_array_helper_c(fmt):
    """Create function to copy contents of a vector.

    Args:
        fmt - format dictionary from the library.
    """
    name = "copy_array"
    if name not in CHelpers:
        helper = dict(
            dependent_helpers=["array_context"],
            c_header="<string.h>",
            cxx_header="<cstring>",
            # Create a single C routine which is called from Fortran via an interface
            # for each cxx_type
            cxx_source=wformat(
                """
// helper function
// Copy std::vector into array c_var(c_var_size).
// Then release std::vector.
void {C_prefix}ShroudCopyArray({C_array_type} *data, \tvoid *c_var, \tsize_t c_var_size)
{{+
const void *cxx_var = data->addr.cvoidp;
int n = c_var_size < data->size ? c_var_size : data->size;
n *= data->len;
{stdlib}memcpy(c_var, cxx_var, n);
{C_memory_dtor_function}(&data->cxx); // delete data->cxx.addr
-}}""",
                fmt,
            ),
        )
        CHelpers[name] = helper


def add_copy_array_helper(fmt):
    """
    Args:
        fmt -
    """
    name = wformat("copy_array_{cxx_type}", fmt)
    if name not in FHelpers:
        helper = dict(
            # XXX when f_kind == C_SIZE_T
            dependent_helpers=["array_context"],
            interface=wformat(
                """
interface+
! helper function
! Copy contents of context into c_var.
subroutine SHROUD_copy_array_{cxx_type}(context, c_var, c_var_size) &+
bind(C, name="{C_prefix}ShroudCopyArray")
use iso_c_binding, only : {f_kind}, C_SIZE_T
import {F_array_type}
type({F_array_type}), intent(IN) :: context
integer({f_kind}), intent(OUT) :: c_var(*)
integer(C_SIZE_T), value :: c_var_size
-end subroutine SHROUD_copy_array_{cxx_type}
-end interface""",
                fmt,
            ),
        )
        FHelpers[name] = helper
    return name


CHelpers = dict(
    ShroudStrCopy=dict(
        c_header="<string.h>",
        c_source="""
// helper function
// Copy src into dest, blank fill to ndest characters
// Truncate if dest is too short.
// dest will not be NULL terminated.
static void ShroudStrCopy(char *dest, int ndest, const char *src, int nsrc)
{
   if (src == NULL) {
     memset(dest,' ',ndest); // convert NULL pointer to blank filled string
   } else {
     if (nsrc < 0) nsrc = strlen(src);
     int nm = nsrc < ndest ? nsrc : ndest;
     memcpy(dest,src,nm);
     if(ndest > nm) memset(dest+nm,' ',ndest-nm); // blank fill
   }
}""",
        cxx_header="<cstring>",
        cxx_source="""
// helper function
// Copy src into dest, blank fill to ndest characters
// Truncate if dest is too short.
// dest will not be NULL terminated.
static void ShroudStrCopy(char *dest, int ndest, const char *src, int nsrc)
{
   if (src == NULL) {
     std::memset(dest,' ',ndest); // convert NULL pointer to blank filled string
   } else {
     if (nsrc < 0) nsrc = std::strlen(src);
     int nm = nsrc < ndest ? nsrc : ndest;
     std::memcpy(dest,src,nm);
     if(ndest > nm) std::memset(dest+nm,' ',ndest-nm); // blank fill
   }
}""",
    ),

    ########################################
    # Used by 'const char *' arguments which need to be NULL terminated
    # in the C wrapper.
    ShroudStrAlloc=dict(
        c_header="<string.h> <stdlib.h>",
        c_source="""
// helper function
// Copy src into new memory and null terminate.
static char *ShroudStrAlloc(const char *src, int nsrc, int ntrim)
{
   char *rv = malloc(nsrc + 1);
   if (ntrim > 0) {
     memcpy(rv, src, ntrim);
   }
   rv[ntrim] = '\\0';
   return rv;
}""",
        cxx_header="<cstring> <cstdlib>",
        cxx_source="""
// helper function
// Copy src into new memory and null terminate.
static char *ShroudStrAlloc(const char *src, int nsrc, int ntrim)
{
   char *rv = (char *) std::malloc(nsrc + 1);
   if (ntrim > 0) {
     std::memcpy(rv, src, ntrim);
   }
   rv[ntrim] = '\\0';
   return rv;
}""",
    ),

    ShroudStrFree=dict(
        c_header="<stdlib.h>",
        c_source="""
// helper function
// Release memory allocated by ShroudStrAlloc
static void ShroudStrFree(char *src)
{
   free(src);
}""",
        cxx_header="<cstdlib>",
        cxx_source="""
// helper function
// Release memory allocated by ShroudStrAlloc
static void ShroudStrFree(char *src)
{
   free(src);
}""",
    ),

    ########################################
    ShroudLenTrim=dict(
        source="""
// helper function
// Returns the length of character string src with length nsrc,
// ignoring any trailing blanks.
int ShroudLenTrim(const char *src, int nsrc) {
    int i;

    for (i = nsrc - 1; i >= 0; i--) {
        if (src[i] != ' ') {
            break;
        }
    }

    return i + 1;
}
"""
    ),
)  # end CHelpers


FHelpers = dict()  # end FHelpers


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
