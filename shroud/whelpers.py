# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
"""Helper functions for C and Fortran wrappers.


 C helper functions which may be added to a implementation file.

 name        = Name of function or type created by the helper.
               This allows the function name to be independent
               of the helper name so that it may include a prefix
               to help control namespace/scope.
               Useful when two helpers create the same function.
               ex. SHROUD_get_from_object_char_{numpy,list}
               Added to the wrapper's format dictionary to allow it to be 
               used in statements.
 api         = "c" or "cxx". Defaults to "c".
               Must be set to "c" for helper functions which will be called
               from Fortran.
               Helpers which use types such as std::string or std::vector
               can only be compiled with C++. Setting api to "c" will add 
               the prototype in an 'extern "C"' block.
 scope       = scope of helper.
               "file" (default) added as file static and may be in
                  several files. source may set source, c_source, or cxx_source.
                  functions must be static since they may be included in 
                  multiple files.
               "cwrap_include" will add to C_header_utility and shared
                  among files. These names need to be unique since they
                  are shared across wrapped libraries.
                  Used with structure and enums.
               "cwrap_impl" - Helpers which are written in C and 
                  called by C or Fortran.
               "pwrap_impl" - Added to PY_utility_filename and shared
                  among files.
 c_include   = List of files to #include
               in implementation file when wrapping a C library.
 cxx_include = List of files to #include.
               in implementation file when wrapping a C++ library.
 c_source    = language=c source.
 cxx_source  = language=c++ source.
 dependent_helpers = list of helpers names needed by this helper
                     They will be added to the output before current helper.
 need_numpy  = If True, NumPy headers will be added.

 proto       = prototype for helper function.
               Must be in the language of api.
 proto_include = List of files to #include before the prototype.
 source      = Code inserted before any wrappers.
               The functions should be file static.
               Used if c_source or cxx_source is not defined.
 include     = Blank delimited list of files to #include.
               Used when c_header and cxx_header are not defined.


 Fortran helper functions which may be added to a module.

 dependent_helpers = list of helpers names needed by this helper
                     They will be added to the output before current helper.
 private   = names for PRIVATE statement
 interface = code for INTERFACE
 source    = code for CONTAINS

# Helper in wrapper classes

Methods in wrappers to deal with helpers.
  add_helper - Build up a list of helpers from statements.
    - wrapf.ModuleInfo.add_f_helper and add_c_helper
    - wrapc.Wrapc.add_c_helper
    - wrapp.Wrapp.add_helper
  gather_helper_code - Write helpers in a sorted order (so the generated
   files will compare). Write dependent helpers so their declaration is before
   their use.

# Fortran helpers

Some Fortran helpers are implemented in C.
Listed in the statements.c_helper and f_helper fields.
The C helpers are written after creating the Fortran wrappers by 
clibrary.write_impl_utility function.

# Python helpers

Most C API functions also return an error indicator, usually NULL if
they are supposed to return a pointer, or -1 if they return an integer.

O& converter - status = converter(PyObject *object, void *address);
The returned status should be 1 for a successful conversion and 0 if
the conversion has failed.

"""

# Note about PRIVATE Fortran helpers
# If a single subroutine uses multiple modules created by Shroud
# some compilers will rightly complain that they each define this function.
#  "Procedure shroud_copy_string_and_free has more than one interface accessible
#  by use association. The interfaces are assumed to be the same."
# It should be marked PRIVATE to prevent users from calling it directly.
# However, gfortran does not like that.
#  "Symbol 'shroud_copy_string_and_free' at (1) is marked PRIVATE but has been given
#  the binding label 'ShroudCopyStringAndFree'"
# See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=49111
# Instead, mangle the name with C_prefix.
# See FHelpers copy_string
#
# This also applies to derived types which are bind(C).


from . import typemap
from . import util

import json

wformat = util.wformat

# Used with literalinclude
# format fields {lstart} and {lend}
cstart = "// start "
cend   = "// end "
fstart = "! start "
fend   = "! end "

_newlibrary = None
def set_library(library):
    global _newlibrary
    _newlibrary = library


def add_all_helpers(symtab):
    """Create helper functions.
    Create helpers for all types.
    """
    fmt = util.Scope(_newlibrary.fmtdict)
    add_external_helpers(symtab)
    add_capsule_helper()
    for ntypemap in symtab.typemaps.values():
        if ntypemap.sgroup == "native":
            add_to_PyList_helper(fmt, ntypemap)
            add_to_PyList_helper_vector(fmt, ntypemap)

def add_external_helpers(symtab):
    """Create helper which have generated names.
    For example, code uses format entries
    C_prefix, C_memory_dtor_function,
    F_array_type, C_array_type

    Some helpers are written in C, but called by Fortran.
    Since the names are external, mangle with C_prefix to avoid
    confict with other Shroud wrapped libraries.

    Args:
        fmtin - format dictionary from the library.
        literalinclude - value of top level option.literalinclude2
    """
    fmtin = _newlibrary.fmtdict
    literalinclude = _newlibrary.options.literalinclude2
    
    fmt = util.Scope(fmtin)
    fmt.lstart = ""
    fmt.lend = ""

    ########################################
    name = "capsule_dtor"
    fmt.hname = name
    fmt.cnamefunc = fmt.C_memory_dtor_function
    fmt.cnameproto = wformat("void {cnamefunc}\t({C_capsule_data_type} *cap)",fmt)
    # Add the C prototype. The body is created Wrapc.write_capsule_code.
    fmt.fnamefunc = wformat("{C_prefix}SHROUD_capsule_dtor", fmt)
    CHelpers[name] = dict(
        name=fmt.cnamefunc,
        api="c",
        dependent_helpers=["capsule_data_helper"],
        proto=fmt.cnameproto + ";",
    )
    FHelpers[name] = dict(
        dependent_helpers=["capsule_data_helper"],
        name=fmt.fnamefunc,
        interface=wformat(
            """
interface+
! helper {hname}
! Delete memory in a capsule.
subroutine {fnamefunc}(ptr)\tbind(C, name="{cnamefunc}")+
import {F_capsule_data_type}
implicit none
type({F_capsule_data_type}), intent(INOUT) :: ptr
-end subroutine {fnamefunc}
-end interface""",
            fmt,
        ),
    )
    
    ########################################
    # XXX - Only used with std::vector and thus C++.
    # Create Fortran interface to helper function
    # which copies an array based on c_type.
    # Each interface calls the same C helper.
    # Used with sgroup="native" types.
    #
    # The function has C_prefix in the name since it is not file static.
    # This allows multiple wrapped libraries to coexist.
    
    name = "copy_array"
    fmt.hname = name
    fmt.cnamefunc = wformat("{C_prefix}ShroudCopyArray", fmt)
    fmt.fnamefunc = wformat("{C_prefix}SHROUD_{hname}", fmt)
    if literalinclude:
        fmt.lstart = "{}helper {}\n".format(cstart, name)
        fmt.lend = "\n{}helper {}".format(cend, name)
    CHelpers[name] = dict(
        name=fmt.cnamefunc,
        scope="cwrap_impl",
        dependent_helpers=["array_context"],
        c_include=["<string.h>", "<stddef.h>"],  # mempcy, size_t
        cxx_include=["<cstring>", "<cstddef>"],
        # Create a single C routine which is called from Fortran
        # via an interface for each cxx_type.
        source=wformat(
                """
{lstart}// helper {hname}
// Copy std::vector into array c_var(c_var_size).
// Then release std::vector.
// Called from Fortran.
void {cnamefunc}({C_array_type} *data, \tvoid *c_var, \tsize_t c_var_size)
{{+
const void *cxx_var = data->base_addr;
int n = c_var_size < data->size ? c_var_size : data->size;
n *= data->elem_len;
{stdlib}memcpy(c_var, cxx_var, n);
-}}{lend}""",
            fmt,
        ),
    )

    FHelpers[name] = dict(
        # XXX when f_kind == C_SIZE_T
        dependent_helpers=["array_context"],
        name=fmt.fnamefunc,
        interface=wformat(
            """
interface+
! helper {hname}
! Copy contents of context into c_var.
subroutine {fnamefunc}(context, c_var, c_var_size) &+
bind(C, name="{cnamefunc}")
use iso_c_binding, only : C_PTR, C_SIZE_T
import {F_array_type}
type({F_array_type}), intent(IN) :: context
type(C_PTR), intent(IN), value :: c_var
integer(C_SIZE_T), value :: c_var_size
-end subroutine {fnamefunc}
-end interface""",
            fmt,
        ),
    )
    
##-    ########################################
##-    # Only used with std::string and thus C++.
##-    name = "string_capsule_size"
##-    fmt.hname = name
##-    if literalinclude:
##-        fmt.lstart = "{}helper {}\n".format(cstart, name)
##-        fmt.lend = "\n{}helper {}".format(cend, name)
##-    fmt.cnamefunc = wformat("{C_prefix}ShroudStringCapsuleSize", fmt)
##-    fmt.fnamefunc = wformat("{C_prefix}SHROUD_string_capsule_size", fmt)
##-    CHelpers[name] = dict(
##-        name=fmt.cnamefunc,
##-        scope="cwrap_impl",
##-        dependent_helpers=["capsule_data_helper"],
##-        cxx_include=["<string>"],
##-        # XXX - mangle name
##-        source=wformat(
##-            """
##-{lstart}// helper {hname}
##-// Extract the length of the std::string in the capsule.
##-// Called by Fortran to deal with allocatable character.
##-size_t {cnamefunc}(\t{C_capsule_data_type} *capsule) {{+
##-const std::string *src = static_cast<const std::string *>(capsule->addr);
##-return src->size();
##--}}{lend}
##-""",
##-            fmt,
##-        ),
##-    )
##-    
##-    # Fortran interface for above function.
##-    # Deal with allocatable character
##-    FHelpers[name] = dict(
##-        dependent_helpers=["capsule_data_helper"],
##-        name=fmt.fnamefunc,
##-        interface=wformat(
##-            """
##-interface+
##-! helper {hname}
##-! Return size of std::string
##-function {fnamefunc}(capsule) &
##-     result(strsize) &
##-     bind(c,name="{cnamefunc}")+
##-use, intrinsic :: iso_c_binding, only : C_CHAR, C_SIZE_T
##-import {F_capsule_data_type}
##-type({F_capsule_data_type}), intent(IN) :: capsule
##-integer(C_SIZE_T) :: strsize
##--end function {fnamefunc}
##--end interface""",
##-            fmt,
##-        ),
##-    )
##-
##-    ##########
##-    name = "copy_string_capsule"
##-    fmt.hname = name
##-    if literalinclude:
##-        fmt.lstart = "{}helper {}\n".format(cstart, name)
##-        fmt.lend = "\n{}helper {}".format(cend, name)
##-    fmt.cnamefunc = wformat("{C_prefix}ShroudCopyStringCapsule", fmt)
##-    fmt.fnamefunc = wformat("{C_prefix}SHROUD_copy_string_capsule", fmt)
##-    CHelpers[name] = dict(
##-        name=fmt.cnamefunc,
##-        scope="cwrap_impl",
##-        dependent_helpers=["capsule_data_helper"],
##-        cxx_include=["<string>", "<cstring>"],
##-        # XXX - mangle name
##-        source=wformat(
##-            """
##-{lstart}// helper {hname}
##-// Copy the char* or std::string in context into c_var.
##-// Called by Fortran to deal with allocatable character.
##-void {cnamefunc}(\t{C_capsule_data_type} *capsule,\t char *c_var,\t size_t c_var_len) {{+
##-const std::string *src = static_cast<const std::string *>(capsule->addr);
##-if (src->empty()) {{+
##-c_var[0] = '\\0';
##--}} else {{+
##-std::strncpy(c_var, src->data(), src->length());
##--}}
##--}}{lend}
##-""",
##-            fmt,
##-        ),
##-    )
##-
##-    # Fortran interface for above function.
##-    # Deal with allocatable character
##-    FHelpers[name] = dict(
##-        dependent_helpers=["capsule_data_helper"],
##-        name=fmt.fnamefunc,
##-        interface=wformat(
##-            """
##-interface+
##-! helper {hname}
##-! Copy the char* or std::string in context into c_var.
##-subroutine {fnamefunc}(capsule, c_var, c_var_size) &
##-     bind(c,name="{cnamefunc}")+
##-use, intrinsic :: iso_c_binding, only : C_CHAR, C_SIZE_T
##-import {F_capsule_data_type}
##-type({F_capsule_data_type}), intent(IN) :: capsule
##-character(kind=C_CHAR), intent(OUT) :: c_var(*)
##-integer(C_SIZE_T), value :: c_var_size
##--end subroutine {fnamefunc}
##--end interface""",
##-            fmt,
##-        ),
##-    )

    ##########
    name = "copy_string"
    fmt.hname = name
    if literalinclude:
        fmt.lstart = "{}helper {}\n".format(cstart, name)
        fmt.lend = "\n{}helper {}".format(cend, name)
    fmt.cnamefunc = wformat("{C_prefix}ShroudCopyString", fmt)
    fmt.fnamefunc = wformat("{C_prefix}SHROUD_copy_string", fmt)
    CHelpers[name] = dict(
        name=fmt.cnamefunc,
        scope="cwrap_impl",
        dependent_helpers=["array_context"],
        cxx_include=["<cstring>", "<cstddef>"],
        # XXX - mangle name
        source=wformat(
            """
{lstart}// helper {hname}
// Copy the char* or std::string in context into c_var.
// Called by Fortran to deal with allocatable character.
void {cnamefunc}(\t{C_array_type} *data,\t char *c_var,\t size_t c_var_len) {{+
const void *cxx_var = data->base_addr;
size_t n = c_var_len;
if (data->elem_len < n) n = data->elem_len;
{stdlib}memcpy(c_var, cxx_var, n);
-}}{lend}
""",
            fmt,
        ),
    )

    # Fortran interface for above function.
    # Deal with allocatable character
    FHelpers[name] = dict(
        dependent_helpers=["array_context"],
        name=fmt.fnamefunc,
        interface=wformat(
            """
interface+
! helper {hname}
! Copy the char* or std::string in context into c_var.
subroutine {fnamefunc}(context, c_var, c_var_size) &
     bind(c,name="{cnamefunc}")+
use, intrinsic :: iso_c_binding, only : C_CHAR, C_SIZE_T
import {F_array_type}
type({F_array_type}), intent(IN) :: context
character(kind=C_CHAR), intent(OUT) :: c_var(*)
integer(C_SIZE_T), value :: c_var_size
-end subroutine {fnamefunc}
-end interface""",
            fmt,
        ),
    )

    ######################################################################
    ########################################
    # std::string *
    ########################################
    # Only used with std::string and thus C++.
    name = "array_string_out"
    fmt.hname = name
    fmt.cnamefunc = wformat("{C_prefix}ShroudArrayStringOut", fmt)
    fmt.cnamefunc_array_string_out = fmt.cnamefunc
    fmt.cnameproto = wformat(
        "void {cnamefunc}({C_array_type} *outdesc, std::string *in, size_t nsize)", fmt)
    if literalinclude:
        fmt.lstart = "{}helper {}\n".format(cstart, name)
        fmt.lend = "\n{}helper {}".format(cend, name)
    CHelpers[name] = dict(
        name=fmt.cnamefunc,
        api="cxx",
        scope="cwrap_impl",
        dependent_helpers=["array_context"],
        proto_include=["<string>", "<vector>"],
        proto=fmt.cnameproto + ";",
        cxx_include=["<cstring>", "<cstddef>"],
        # XXX - mangle name
        source=wformat(
            """
{lstart}// helper {hname}
// Copy the std::vector<std::string> into Fortran array argument.
// Called by C++.
{cnameproto}
{{+
size_t nvect = outdesc->size;
size_t len = outdesc->elem_len;
char *dest = static_cast<char *>(outdesc->base_addr);
// Clear user memory
std::memset(dest, ' ', nvect*len);

// Copy into user memory
nvect = std::min(nvect, nsize);
for (size_t i = 0; i < nvect; ++i) {{+
std::memcpy(dest, in[i].data(), std::min(len, in[i].length()));
dest += outdesc->elem_len;
-}}
-}}{lend}
""",
            fmt,
        ),
    )

    # Fortran interface for above function.
    # Deal with allocatable character
##-    fmt.hnamefunc = wformat("{C_prefix}SHROUD_copy_array_string_and_free", fmt)
##-    FHelpers[name] = dict(
##-        dependent_helpers=["array_context"],
##-        name=fmt.hnamefunc,
##-        interface=wformat(
##-            """
##-interface+
##-! helper {hname}
##-! Copy the char* or std::string in context into c_var.
##-subroutine {hnamefunc}(context, c_var, c_var_size) &
##-     bind(c,name="{C_prefix}ShroudCopyStringAndFree")+
##-use, intrinsic :: iso_c_binding, only : C_CHAR, C_SIZE_T
##-import {F_array_type}
##-type({F_array_type}), intent(IN) :: context
##-character(kind=C_CHAR), intent(OUT) :: c_var(*)
##-integer(C_SIZE_T), value :: c_var_size
##--end subroutine {hnamefunc}
##--end interface""",
##-            fmt,
##-        ),
##-    )

    ########################################
    ########################################
    # Only used with std::string and thus C++.
    # Called from Fortran.
    # The capsule contains a pointer to a std::vector<std::string>
    # which is copied into the cdesc.
    name = "array_string_allocatable"
    fmt.hname = name
    fmt.cnamefunc = wformat("{C_prefix}ShroudArrayStringAllocatable", fmt)
    fmt.fnamefunc = wformat("{C_prefix}SHROUD_array_string_allocatable", fmt)
    fmt.cnameproto = wformat(
        "void {cnamefunc}({C_array_type} *dest, {C_capsule_data_type} *src)", fmt)
    if literalinclude:
        fmt.lstart = "{}helper {}\n".format(cstart, name)
        fmt.lend = "\n{}helper {}".format(cend, name)
    CHelpers[name] = dict(
        name=fmt.cnamefunc,
        api="c",
        scope="cwrap_impl",
        dependent_helpers=["capsule_data_helper", "array_context", "array_string_out"],
        proto=fmt.cnameproto + ";",
        source=wformat(
            """
{lstart}// helper {hname}
// Copy the std::string array into Fortran array.
// Called by Fortran to deal with allocatable character.
// out is already blank filled.
{cnameproto}
{{+
std::string *cxxvec =\t static_cast< std::string *>\t(src->addr);
{cnamefunc_array_string_out}(dest, cxxvec, dest->size);
-}}{lend}
""",
            fmt,
        ),
    )

    # Fortran interface for above function.
    # Deal with allocatable character
    FHelpers[name] = dict(
        dependent_helpers=["array_context"],
        name=fmt.fnamefunc,
        interface=wformat(
            """
interface+
! helper {hname}
subroutine {fnamefunc}(dest, src) &
     bind(c,name="{cnamefunc}")+
import {F_array_type}, {F_capsule_data_type}
type({F_array_type}), intent(IN) :: dest
type({F_capsule_data_type}), intent(IN) :: src
-end subroutine {fnamefunc}
-end interface""",
            fmt,
        ),
    )

    ########################################
    ########################################
    name = "array_string_out_len"
    fmt.hname = name
    fmt.cnamefunc = wformat("{C_prefix}ShroudArrayStringOutSize", fmt)
    fmt.cnameproto = wformat(
        "size_t {cnamefunc}(std::string *in, size_t nsize)", fmt)
    if literalinclude:
        fmt.lstart = "{}helper {}\n".format(cstart, name)
        fmt.lend = "\n{}helper {}".format(cend, name)
    CHelpers[name] = dict(
        name=fmt.cnamefunc,
        api="cxx",
        scope="cwrap_impl",
        proto_include=["<string>", "<vector>"],
        proto=fmt.cnameproto + ";",
        source=wformat(
            """
{lstart}// helper {hname}
// Return the maximum string length in a std::vector<std::string>.
{cnameproto}
{{+
size_t len = 0;
for (size_t i = 0; i < nsize; ++i) {{+
len = std::max(len, in[i].length());
-}}
return len;
-}}{lend}
""",
            fmt,
        ),
    )

    ########################################



    ######################################################################
    ########################################
    #   std::vector< std::string >
    ########################################
    # Only used with std::string and thus C++.
    name = "vector_string_out"
    fmt.hname = name
    fmt.cnamefunc = wformat("{C_prefix}ShroudVectorStringOut", fmt)
    fmt.fnamefunc = wformat("{C_prefix}shroud_vector_string_out", fmt)
    fmt.cnamefunc_vector_string_out = fmt.cnamefunc
    fmt.cnameproto = wformat(
        "void {cnamefunc}({C_array_type} *outdesc, std::vector<std::string> &in)", fmt)
    if literalinclude:
        fmt.lstart = "{}helper {}\n".format(cstart, name)
        fmt.lend = "\n{}helper {}".format(cend, name)
    CHelpers[name] = dict(
        name=fmt.cnamefunc,
        api="cxx",
        scope="cwrap_impl",
        dependent_helpers=["array_context"],
        proto_include=["<string>", "<vector>"],
        proto=fmt.cnameproto + ";",
        cxx_include=["<cstring>", "<cstddef>"],
        # XXX - mangle name
        source=wformat(
            """
{lstart}// helper {hname}
// Copy the std::vector<std::string> into Fortran array argument.
// Called by C++.
{cnameproto}
{{+
size_t nvect = outdesc->size;
size_t len = outdesc->elem_len;
char *dest = static_cast<char *>(outdesc->base_addr);
// Clear user memory
std::memset(dest, ' ', nvect*len);

// Copy into user memory
nvect = std::min(nvect, in.size());
//char *dest = static_cast<char *>(outdesc->cxx.addr);
for (size_t i = 0; i < nvect; ++i) {{+
std::memcpy(dest, in[i].data(), std::min(len, in[i].length()));
dest += outdesc->elem_len;
-}}
-}}{lend}
""",
            fmt,
        ),
    )

    # Fortran interface for above function.
    FHelpers[name] = dict(
        dependent_helpers=["array_context"],
        name=fmt.fnamefunc,
        interface=wformat(
            """
interface+
! helper {hname}
subroutine {fnamefunc}(out, in) &
     bind(c,name="{cnamefunc}")+
use, intrinsic :: iso_c_binding, only : C_PTR
import {F_array_type}
type({F_array_type}), intent(IN) :: out
type(C_PTR), intent(IN) :: in
-end subroutine {fnamefunc}
-end interface""",
            fmt,
        ),
    )

    ########################################

    # Fortran interface for above function.
    # Deal with allocatable character
##-    fmt.hnamefunc = wformat("{C_prefix}SHROUD_copy_vector_string_and_free", fmt)
##-    FHelpers[name] = dict(
##-        dependent_helpers=["array_context"],
##-        name=fmt.hnamefunc,
##-        interface=wformat(
##-            """
##-interface+
##-! helper {hname}
##-! Copy the char* or std::string in context into c_var.
##-subroutine {hnamefunc}(context, c_var, c_var_size) &
##-     bind(c,name="{C_prefix}ShroudCopyStringAndFree")+
##-use, intrinsic :: iso_c_binding, only : C_CHAR, C_SIZE_T
##-import {F_array_type}
##-type({F_array_type}), intent(IN) :: context
##-character(kind=C_CHAR), intent(OUT) :: c_var(*)
##-integer(C_SIZE_T), value :: c_var_size
##--end subroutine {hnamefunc}
##--end interface""",
##-            fmt,
##-        ),
##-    )

    ########################################
    ########################################
    # Only used with std::string and thus C++.
    # Called from Fortran.
    # The capsule contains a pointer to a std::vector<std::string>
    # which is copied into the cdesc.
    name = "vector_string_allocatable"
    fmt.hname = name
    fmt.cnamefunc = wformat("{C_prefix}ShroudVectorStringAllocatable", fmt)
    fmt.fnamefunc = wformat("{C_prefix}SHROUD_vector_string_allocatable", fmt)
    fmt.cnameproto = wformat(
        "void {cnamefunc}({C_array_type} *dest, {C_capsule_data_type} *src)", fmt)
    if literalinclude:
        fmt.lstart = "{}helper {}\n".format(cstart, name)
        fmt.lend = "\n{}helper {}".format(cend, name)
    CHelpers[name] = dict(
        name=fmt.cnamefunc,
        api="c",
        scope="cwrap_impl",
        dependent_helpers=["capsule_data_helper", "array_context", "vector_string_out"],
        proto=fmt.cnameproto + ";",
        source=wformat(
            """
{lstart}// helper {hname}
// Copy the std::vector<std::string> into Fortran array.
// Called by Fortran to deal with allocatable character.
// out is already blank filled.
{cnameproto}
{{+
std::vector<std::string> *cxxvec =\t static_cast< std::vector<std::string> * >\t(src->addr);
{cnamefunc_vector_string_out}(dest, *cxxvec);
-}}{lend}
""",
            fmt,
        ),
    )

    # Fortran interface for above function.
    # Deal with allocatable character
    FHelpers[name] = dict(
        dependent_helpers=["array_context"],
        name=fmt.fnamefunc,
        interface=wformat(
            """
interface+
! helper {hname}
! Copy the char* or std::string in context into c_var.
subroutine {fnamefunc}(dest, src) &
     bind(c,name="{cnamefunc}")+
import {F_capsule_data_type}, {F_array_type}
type({F_array_type}), intent(IN) :: dest
type({F_capsule_data_type}), intent(IN) :: src
-end subroutine {fnamefunc}
-end interface""",
            fmt,
        ),
    )

    ########################################
    ########################################
    name = "vector_string_out_len"
    fmt.hname = name
    fmt.cnamefunc = wformat("{C_prefix}ShroudVectorStringOutSize", fmt)
    fmt.cnameproto = wformat(
        "size_t {cnamefunc}(std::vector<std::string> &in)", fmt)
    if literalinclude:
        fmt.lstart = "{}helper {}\n".format(cstart, name)
        fmt.lend = "\n{}helper {}".format(cend, name)
    CHelpers[name] = dict(
        name=fmt.cnamefunc,
        api="cxx",
        scope="cwrap_impl",
        proto_include=["<string>", "<vector>"],
        proto=fmt.cnameproto + ";",
        source=wformat(
            """
{lstart}// helper {hname}
// Return the maximum string length in a std::vector<std::string>.
{cnameproto}
{{+
size_t nvect = in.size();
size_t len = 0;
for (size_t i = 0; i < nvect; ++i) {{+
len = std::max(len, in[i].length());
-}}
return len;
-}}{lend}
""",
            fmt,
        ),
    )

    ######################################################################
    ########################################
    ########################################
    name = "pointer_string"
    # Set Fortran POINTER to string.
    # Must be a function (or a F2008 BLOCK) since fptr must
    # be declared after the string length is known.
    fmt.hname = name
    fmt.fnamefunc = wformat("{C_prefix}SHROUD_pointer_string", fmt)
    FHelpers[name] = dict(
        dependent_helpers=["array_context"],
        name=fmt.fnamefunc,
        source=wformat(
            """
! helper {hname}
! Assign context to an assumed-length character pointer
subroutine {fnamefunc}(context, var)+
use iso_c_binding, only : c_f_pointer, C_PTR
implicit none
type({F_array_type}), intent(IN) :: context
character(len=:), pointer, intent(OUT) :: var
character(len=context%elem_len), pointer :: fptr
call c_f_pointer(context%base_addr, fptr)
var => fptr
-end subroutine {fnamefunc}""",
            fmt,
        ),
    )
    
    ########################################
    name = "string_to_cdesc"
    fmt.hname = name
    if literalinclude:
        fmt.lstart = "{}helper {}\n".format(cstart, name)
        fmt.lend = "\n{}helper {}".format(cend, name)
    fmt.cnamefunc = "ShroudStringToCdesc"
    CHelpers[name] = dict(
        name=fmt.cnamefunc,
        dependent_helpers=["array_context"],
        cxx_include=["<cstring>", "<cstddef>"],
        source=wformat(
            """
{lstart}// helper {hname}
// Save std::string metadata into array to allow Fortran to access values.
// CHARACTER(len=elem_size) src
static void {cnamefunc}(\t{C_array_type} *cdesc,\t const std::string * src)
{{+
if (src->empty()) {{+
cdesc->base_addr = NULL;
cdesc->elem_len = 0;
-}} else {{+
cdesc->base_addr =\t const_cast<char *>(\tsrc->data());
cdesc->elem_len = src->length();
-}}
cdesc->size = 1;
cdesc->rank = 0;  // scalar
-}}{lend}""", fmt),
    )

    
    ########################################
    # Python
    ########################################
    name = "py_capsule_dtor"
    fmt.hname = name
    fmt.hnamefunc = wformat("FREE_{hname}", fmt)
    CHelpers[name] = dict(
        name=fmt.hnamefunc,
        source=wformat(
            """
// helper {hname}
// Release memory in PyCapsule.
// Used with native arrays.
static void {hnamefunc}(PyObject *obj)
{{+
void *in = PyCapsule_GetPointer(obj, {nullptr});
if (in != {nullptr}) {{+
{stdlib}free(in);
-}}
-}}""",
            fmt,
        ),
    )
    
    ########################################
    # char *
    name = "get_from_object_char"
    fmt.hname = name
    fmt.hnamefunc = fmt.PY_helper_prefix + name
    fmt.hnameproto = wformat(
            "int {hnamefunc}\t(PyObject *obj,\t {PY_typedef_converter} *value)", fmt)
    CHelpers[name] = dict(
        name=fmt.hnamefunc,
        dependent_helpers=["PY_converter_type"],
        proto=fmt.hnameproto + ";",
        source=wformat("""
// helper {hname}
// Converter from PyObject to char *.
// The returned status will be 1 for a successful conversion
// and 0 if the conversion has failed.
// value.obj is unused.
// value.dataobj - object which holds the data.
// If same as obj argument, its refcount is incremented.
// value.data is owned by value.dataobj and must be copied to be preserved.
// Caller must use Py_XDECREF(value.dataobj).
{PY_helper_static}{hnameproto}
{{+
size_t size = 0;
char *out;
if (PyUnicode_Check(obj)) {{+
^#if PY_MAJOR_VERSION >= 3
PyObject *strobj = PyUnicode_AsUTF8String(obj);
out = PyBytes_AS_STRING(strobj);
size = PyBytes_GET_SIZE(strobj);
value->dataobj = strobj;  // steal reference
^#else
PyObject *strobj = PyUnicode_AsUTF8String(obj);
out = PyString_AsString(strobj);
size = PyString_Size(obj);
value->dataobj = strobj;  // steal reference
^#endif
^#if PY_MAJOR_VERSION < 3
-}} else if (PyString_Check(obj)) {{+
out = PyString_AsString(obj);
size = PyString_Size(obj);
value->dataobj = obj;
Py_INCREF(obj);
^#endif
-}} else if (PyBytes_Check(obj)) {{+
out = PyBytes_AS_STRING(obj);
size = PyBytes_GET_SIZE(obj);
value->dataobj = obj;
Py_INCREF(obj);
-}} else if (PyByteArray_Check(obj)) {{+
out = PyByteArray_AS_STRING(obj);
size = PyByteArray_GET_SIZE(obj);
value->dataobj = obj;
Py_INCREF(obj);
-}} else if (obj == Py_None) {{+
out = NULL;
size = 0;
value->dataobj = NULL;
-}} else {{+
PyErr_Format(PyExc_TypeError,\t "argument should be string or None, not %.200s",\t Py_TYPE(obj)->tp_name);
return 0;
-}}
value->obj = {nullptr};
value->data = out;
value->size = size;
return 1;
-}}
""", fmt),
    )
    # There are no 'list' or 'numpy' version of these functions.
    # Use the one-true-version get_from_object_char.
    CHelpers['get_from_object_char_list'] = dict(
        name=fmt.hnamefunc,
        dependent_helpers=[name],
    )
    CHelpers['get_from_object_char_numpy'] = dict(
        name=fmt.hnamefunc,
        dependent_helpers=[name],
    )

    ##########
    # Generate C or C++ version of helper.
    ##########
    # 'char *' needs a custom handler because of the nature
    # of NULL terminated strings.
    ntypemap = symtab.lookup_typemap("char")
    fmt.fcn_suffix = "char"
    fmt.fcn_type = "string"
    fmt.c_type = "char *"
    fmt.Py_ctor = ntypemap.PY_ctor.format(ctor_expr="in[i]")
    fmt.c_const=""  # XXX issues with struct.yaml test, remove const.
    fmt.hname = "to_PyList_char"
    CHelpers["to_PyList_char"] = create_to_PyList(fmt)

    ########################################
    name = "fill_from_PyObject_char"
    fmt.hname = name
    fmt.hnamefunc = fmt.PY_helper_prefix + name
    fmt.hnameproto = wformat(
            "int {hnamefunc}\t(PyObject *obj,\t const char *name,\t char *in,\t Py_ssize_t insize)", fmt)
    CHelpers[name] = dict(
        name=fmt.hnamefunc,
        dependent_helpers=["get_from_object_char"],
        c_include=["<string.h>"],
        cxx_include=["<cstring>"],
        proto=fmt.hnameproto + ";",
        source=wformat("""
// helper {hname}
// Fill existing char array from PyObject.
// Return 0 on success, -1 on error.
{PY_helper_static}{hnameproto}
{{+
{PY_typedef_converter} value;
int i = {PY_helper_prefix}get_from_object_char(obj, &value);
if (i == 0) {{+
Py_DECREF(obj);
return -1;
-}}
if (value.data == {nullptr}) {{+
in[0] = '\\0';
-}} else {{+
{stdlib}strncpy\t(in,\t {cast_static}char *{cast1}value.data{cast2},\t insize);
Py_DECREF(value.dataobj);
-}}
return 0;
-}}""", fmt),
    )

    ########################################
    # char **
    name = "get_from_object_charptr"
    fmt.size_var="size"
    fmt.c_var="in"
    fmt.hname = name
    fmt.hnamefunc = fmt.PY_helper_prefix + name
    CHelpers[name] = create_get_from_object_list_charptr(fmt)
    # There are no 'list' or 'numpy' version of these functions.
    # Use the one-true-version SHROUD_get_from_object_charptr.
    CHelpers['get_from_object_charptr_list'] = dict(
        name=fmt.hnamefunc,
        dependent_helpers=[name],
    )
    CHelpers['get_from_object_charptr_numpy'] = dict(
        name=fmt.hnamefunc,
        dependent_helpers=[name],
    )

    ########################################
    CHelpers['PY_converter_type'] = dict(
        scope="pwrap_impl",
        c_include=["<stddef.h>"],
        cxx_include=["<cstddef>"],
        # obj may be the argument passed into a function or
        # it may be a PyCapsule for locally allocated memory.
        source=wformat("""
// helper PY_converter_type
// Store PyObject and pointer to the data it contains.
// name - used in error messages
// obj  - A mutable object which holds the data.
//        For example, a NumPy array, Python array.
//        But not a list or str object.
// dataobj - converter allocated memory.
//           Decrement dataobj to release memory.
//           For example, extracted from a list or str.
// data  - C accessable pointer to data which is in obj or dataobj.
// size  - number of items in data (not number of bytes).
typedef struct {{+
const char *name;
PyObject *obj;
PyObject *dataobj;
void *data;   // points into obj.
size_t size;
-}} {PY_typedef_converter};""", fmt)
    )
    
######################################################################

def add_capsule_helper():
    """Share info with C++ to allow Fortran to release memory.

    Used with shadow classes and std::vector.
    """
    fmtin = _newlibrary.fmtdict
    literalinclude = _newlibrary.options.literalinclude2
    # Add some format strings
    fmt = util.Scope(fmtin)
    name = "capsule_data_helper"
    fmt.hname = name
    if literalinclude:
        fmt.lstart = "{}helper {}\n".format(fstart, name)
        fmt.lend = "\n{}helper {}".format(fend, name)
    else:
        fmt.lstart = ""
        fmt.lend = ""

    helper = dict(
        name=fmt.F_capsule_data_type,
        derived_type=wformat(
            """
{lstart}! helper {hname}
type, bind(C) :: {F_capsule_data_type}+
type(C_PTR) :: addr = C_NULL_PTR  ! address of C++ memory
integer(C_INT) :: idtor = 0       ! index of destructor
-end type {F_capsule_data_type}{lend}""",
            fmt,
        ),
        modules=dict(iso_c_binding=["C_PTR", "C_INT", "C_NULL_PTR"]),
    )
    FHelpers[name] = helper

    helper = dict(
        scope="cwrap_include",
        source=wformat(
            """
// helper {hname}
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
    fmt.hname = name
    fmt.__helper = FHelpers["capsule_dtor"]["name"]
    # XXX split helper into to parts, one for each derived type
    helper = dict(
        dependent_helpers=["capsule_data_helper", "capsule_dtor"],
        derived_type=wformat(
            """
! helper {hname}
type :: {F_capsule_type}+
private
type({F_capsule_data_type}) :: mem
-contains
+final :: {F_capsule_final_function}
procedure :: delete => {F_capsule_delete_function}
-end type {F_capsule_type}""",
            fmt,
        ),
        # cannot be declared with both PRIVATE and BIND(C) attributes
        source=wformat(
            """
! helper {hname}
! finalize a static {F_capsule_data_type}
subroutine {F_capsule_final_function}(cap)+
type({F_capsule_type}), intent(INOUT) :: cap
call {__helper}(cap%mem)
-end subroutine {F_capsule_final_function}

subroutine {F_capsule_delete_function}(cap)+
class({F_capsule_type}) :: cap
call {__helper}(cap%mem)
-end subroutine {F_capsule_delete_function}""",
            fmt,
        ),
    )
    FHelpers[name] = helper

    ########################################
    name = "array_context"
    fmt.hname = name
    if literalinclude:
        fmt.lstart = "{}{}\n".format(cstart, name)
        fmt.lend = "\n{}{}".format(cend, name)
    helper = dict(
        name=fmt.C_array_type,
        scope="cwrap_include",
        include=["<stddef.h>"],
        # Create a union for addr to avoid some casts.
        # And help with debugging since ccharp will display contents.
        source=wformat(
            """
{lstart}// helper {hname}
struct s_{C_array_type} {{+
void * base_addr;
int type;        /* type of element */
size_t elem_len; /* bytes-per-item or character len in c++ */
size_t size;     /* size of data in c++ */
int rank;        /* number of dimensions, 0=scalar */
long shape[7];
-}};
typedef struct s_{C_array_type} {C_array_type};{lend}""",
            fmt,
        ),
        dependent_helpers=["type_defines"], # used with type field
    )
    CHelpers[name] = helper

    # Create a derived type used to communicate with C wrapper.
    # Should never be exposed to user.
    # Inspired by futher interoperability with C.
    # XXX - shape is C_LONG, maybe it should be C_PTRDIFF_T.
    if literalinclude:
        fmt.lstart = "{}{}\n".format(fstart, name)
        fmt.lend = "\n{}{}".format(fend, name)
    helper = dict(
        name=fmt.F_array_type,
        derived_type=wformat(
            """
{lstart}! helper {hname}
type, bind(C) :: {F_array_type}+
! address of data
type(C_PTR) :: base_addr = C_NULL_PTR
! type of element
integer(C_INT) :: type
! bytes-per-item or character len of data in cxx
integer(C_SIZE_T) :: elem_len = 0_C_SIZE_T
! size of data in cxx
integer(C_SIZE_T) :: size = 0_C_SIZE_T
! number of dimensions
integer(C_INT) :: rank = -1
integer(C_LONG) :: shape(7) = 0
-end type {F_array_type}{lend}""",
            fmt,
        ),
        modules=dict(iso_c_binding=[
            "C_NULL_PTR", "C_PTR", "C_SIZE_T", "C_INT", "C_LONG"]),
        dependent_helpers=["type_defines"], # used with type field
    )
    FHelpers[name] = helper


def add_to_PyList_helper(fmt, ntypemap):
    """Add helpers to work with Python lists.
    Several helpers are created based on the type of arg.
    Used with sgroup="native" types.

    Args:
        fmt      - util.Scope, parent is newlibrary
        ntypemap - typemap.Typemap
    """
    flat_name = ntypemap.flat_name
    fmt.c_type = ntypemap.c_type
    fmt.numpy_type = ntypemap.PYN_typenum

    ########################################
    # Used with intent(out)
    name = "to_PyList_" + flat_name
    if ntypemap.PY_ctor is not None:
        fmt.hname = name
        fmt.fcn_suffix = flat_name
        ctor_expr = "in[i]"
        if ntypemap.py_ctype is not None:
            ctor_expr = ntypemap.pytype_to_pyctor.format(ctor_expr=ctor_expr)
        fmt.Py_ctor = ntypemap.PY_ctor.format(ctor_expr=ctor_expr)
        fmt.c_const="const "
        helper = create_to_PyList(fmt)
        CHelpers[name] = create_to_PyList(fmt)

    ########################################
    # Used with intent(inout)
    name = "update_PyList_" + flat_name
    if ntypemap.PY_ctor is not None:
        ctor_expr = "in[i]"
        if ntypemap.py_ctype is not None:
            ctor_expr = ntypemap.pytype_to_pyctor.format(ctor_expr=ctor_expr)
        fmt.Py_ctor = ntypemap.PY_ctor.format(ctor_expr=ctor_expr)
        fmt.hname = name
        fmt.hnameproto = wformat(
            "void {PY_helper_prefix}{hname}\t(PyObject *out, {c_type} *in, size_t size)", fmt)
        helper = dict(
            proto=fmt.hnameproto + ";",
            source=wformat(
                """
// helper {hname}
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
{PY_helper_static}{hnameproto}
{{+
for (size_t i = 0; i < size; ++i) {{+
PyObject *item = PyList_GET_ITEM(out, i);
Py_DECREF(item);
PyList_SET_ITEM(out, i, {Py_ctor});
-}}
-}}""", fmt),
        )
        CHelpers[name] = helper

    ########################################
    # Used with intent(in), setter.
    # Return -1 on error.
    # Use a fixed text in PySequence_Fast.
    # If an error occurs, replace message with one which includes argument name.
    if ntypemap.PY_get:
        name = "fill_from_PyObject_" + flat_name + "_list"
        fmt.hname = name
        fmt.flat_name = flat_name
        fmt.fcn_type = ntypemap.c_type
        fmt.py_ctype = fmt.c_type
        fmt.work_ctor = "cvalue"
        if ntypemap.py_ctype is not None:
            fmt.py_ctype = ntypemap.py_ctype
            fmt.work_ctor = ntypemap.pytype_to_cxx.format(work_var=fmt.work_ctor)
        fmt.Py_get_obj = ntypemap.PY_get.format(py_var="obj")
        fmt.Py_get = ntypemap.PY_get.format(py_var="item")
        CHelpers[name] = fill_from_PyObject_list(fmt)

        name = "fill_from_PyObject_" + flat_name + "_numpy"
        fmt.hname = name
        CHelpers[name] = fill_from_PyObject_numpy(fmt)

    ########################################
    # Function called by typemap.PY_get_converter for NumPy.
    name = "get_from_object_{}_numpy".format(flat_name)
    fmt.py_tmp = "array"
    fmt.c_type = ntypemap.c_type
    fmt.numpy_type = ntypemap.PYN_typenum
    fmt.hname = name
    fmt.hnamefunc = fmt.PY_helper_prefix + name
    fmt.hnameproto = wformat(
        "int {hnamefunc}\t(PyObject *obj,\t {PY_typedef_converter} *value)", fmt)
    helper = dict(
        name=fmt.hnamefunc,
        dependent_helpers=["PY_converter_type"],
        need_numpy=True,
        proto=fmt.hnameproto + ";",
        source=wformat("""
// helper {hname}
// Convert PyObject to {c_type} pointer.
{PY_helper_static}{hnameproto}
{{+
PyObject *{py_tmp} = PyArray_FROM_OTF(obj,\t {numpy_type},\t NPY_ARRAY_IN_ARRAY);
if ({py_tmp} == {nullptr}) {{+
PyErr_SetString(PyExc_ValueError,\t "must be a 1-D array of {c_type}");
return 0;
-}}
value->obj = {py_tmp};
value->dataobj = {nullptr};
value->data = PyArray_DATA({cast_reinterpret}PyArrayObject *{cast1}{py_tmp}{cast2});
value->size = PyArray_SIZE({cast_reinterpret}PyArrayObject *{cast1}{py_tmp}{cast2});
return 1;
-}}""", fmt),
    )
    CHelpers[name] = helper

    ########################################
    # Function called by typemap.PY_get_converter for list.
    if ntypemap.PY_get:
        name = "get_from_object_{}_list".format(flat_name)
        fmt.size_var = "size"
        fmt.c_var = "in"
        fmt.fcn_suffix = flat_name
        fmt.Py_get = ntypemap.PY_get.format(py_var="item")
        fmt.hname = name
        fmt.hnamefunc = fmt.PY_helper_prefix + name
        CHelpers[name] = create_get_from_object_list(fmt)

def fill_from_PyObject_list(fmt):
    """Create helper to convert list of PyObjects to existing C array.

    If passed a scalar, broadcast to array.
    """
    fmt.hnamefunc = wformat(
        "{PY_helper_prefix}fill_from_PyObject_{flat_name}_list", fmt)
    fmt.hnameproto = wformat(
            "int {hnamefunc}\t(PyObject *obj,\t const char *name,\t "
            "{c_type} *in,\t Py_ssize_t insize)", fmt)
    helper = dict(
        name=fmt.hnamefunc,
        proto=fmt.hnameproto + ";",
        source=wformat(
                """
// helper {hname}
// Fill {c_type} array from Python sequence object.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
{PY_helper_static}{hnameproto}
{{+
{py_ctype} cvalue = {Py_get_obj};
if (!PyErr_Occurred()) {{+
// Broadcast scalar.
for (Py_ssize_t i = 0; i < insize; ++i) {{+
in[i] = {work_ctor};
-}}
return 0;
-}}
PyErr_Clear();

// Look for sequence.
PyObject *seq = PySequence_Fast(obj, "holder");
if (seq == NULL) {{+
PyErr_Format(PyExc_TypeError,\t "argument '%s' must be iterable",\t name);
return -1;
-}}
Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
if (size > insize) {{+
size = insize;
-}}
for (Py_ssize_t i = 0; i < size; ++i) {{+
PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
cvalue = {Py_get};
if (PyErr_Occurred()) {{+
Py_DECREF(seq);
PyErr_Format(PyExc_TypeError,\t "argument '%s', index %d must be {fcn_type}",\t name,\t (int) i);
return -1;
-}}
in[i] = {work_ctor};
-}}
Py_DECREF(seq);
return 0;
-}}""", fmt),
    )
    return helper
    
def fill_from_PyObject_numpy(fmt):
    """Create helper to convert list of PyObjects to existing C array.

    If passed a scalar, broadcast to array.
    """
    fmt.hnamefunc = wformat(
        "{PY_helper_prefix}fill_from_PyObject_{flat_name}_numpy", fmt)
    fmt.hnameproto = wformat(
            "int {hnamefunc}\t(PyObject *obj,\t const char *name,\t {c_type} *in,\t Py_ssize_t insize)", fmt)
    fmt.py_tmp = "array"
    fmt.numpy_type
    helper = dict(
        name=fmt.hnamefunc,
        proto=fmt.hnameproto + ";",
        need_numpy=True,
        source=wformat(
                """
// helper {hname}
// Fill {c_type} array from Python object using NumPy.
// If obj is a scalar, broadcast to array.
// Return 0 on success, -1 on error.
{PY_helper_static}{hnameproto}
{{+
{py_ctype} cvalue = {Py_get_obj};
if (!PyErr_Occurred()) {{+
// Broadcast scalar.
for (Py_ssize_t i = 0; i < insize; ++i) {{+
in[i] = {work_ctor};
-}}
return 0;
-}}
PyErr_Clear();

PyObject *{py_tmp} = PyArray_FROM_OTF(obj,\t {numpy_type},\t NPY_ARRAY_IN_ARRAY);
if ({py_tmp} == {nullptr}) {{+
PyErr_Format(PyExc_TypeError,\t "argument '%s' must be a 1-D array of {c_type}",\t name);
return -1;
-}}
PyArrayObject *pyarray = {cast_reinterpret}PyArrayObject *{cast1}{py_tmp}{cast2};

{c_type} *data = {cast_static}{c_type} *{cast1}PyArray_DATA(pyarray){cast2};
npy_intp size = PyArray_SIZE(pyarray);
if (size > insize) {{+
size = insize;
-}}
for (Py_ssize_t i = 0; i < size; ++i) {{+
in[i] = data[i];
-}}
Py_DECREF(pyarray);
return 0;
-}}""", fmt),
        )
    return helper
    
def create_to_PyList(fmt):
    """Create helper to convert C array to PyList of PyObjects.
    """
    fmt.hnamefunc = wformat(
        "{PY_helper_prefix}to_PyList_{fcn_suffix}", fmt)
    fmt.hnameproto = wformat(
        "PyObject *{hnamefunc}\t({c_const}{c_type} *in, size_t size)", fmt)
    helper = dict(
        name=fmt.hnamefunc,
        proto=fmt.hnameproto + ";",
        source=wformat(
            """
// helper {hname}
// Convert {c_type} pointer to PyList of PyObjects.
{PY_helper_static}{hnameproto}
{{+
PyObject *out = PyList_New(size);
for (size_t i = 0; i < size; ++i) {{+
PyList_SET_ITEM(out, i, {Py_ctor});
-}}
return out;
-}}""", fmt),
    )
    return helper

def create_get_from_object_list(fmt):
    """ Convert PyObject to {c_type} pointer.
    Used with native types.
# XXX - convert empty list to NULL pointer.

    format fields:
       fcn_suffix - 
    """
    fmt.hnameproto = wformat(
            "int {hnamefunc}\t(PyObject *obj,\t {PY_typedef_converter} *value)", fmt)
    fmt.dtor_helper = CHelpers["py_capsule_dtor"]["name"]
    helper = dict(
        name=fmt.hnamefunc,
        dependent_helpers=[
            "PY_converter_type",
            "py_capsule_dtor",
        ],
        c_include=["<stdlib.h>"],   # malloc/free
        cxx_include=["<cstdlib>"],  # malloc/free
        proto=fmt.hnameproto + ";",
        source=wformat("""
// helper {hname}
// Convert list of PyObject to array of {c_type}.
// Return 0 on error, 1 on success.
// Set Python exception on error.
{PY_helper_static}{hnameproto}
{{+
PyObject *seq = PySequence_Fast(obj, "holder");
if (seq == NULL) {{+
PyErr_Format(PyExc_TypeError,\t "argument '%s' must be iterable",\t value->name);
return 0;
-}}
Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
{c_type} *in = {cast_static}{c_type} *{cast1}{stdlib}malloc(size * sizeof({c_type})){cast2};
for (Py_ssize_t i = 0; i < size; i++) {{+
PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
{c_type} cvalue = {Py_get};
if (PyErr_Occurred()) {{+
{stdlib}free(in);
Py_DECREF(seq);
PyErr_Format(PyExc_TypeError,\t "argument '%s', index %d must be {fcn_type}",\t value->name,\t (int) i);
return 0;
-}}
in[i] = {work_ctor};
-}}
Py_DECREF(seq);

value->obj = {nullptr};  // Do not save list object.
value->dataobj = PyCapsule_New(in, {nullptr}, {dtor_helper});
value->data = {cast_static}{c_type} *{cast1}{c_var}{cast2};
value->size = size;
return 1;
-}}""", fmt),
    )
    return helper

def create_get_from_object_list_charptr(fmt):
    """ Convert PyObject to an char **.
    ["one", "two"]
    helper get_from_object_charptr

    Loop over all strings in the sequence object and
    convert using get_from_object_char helper
    which deals with unicode.
    All string values are copied into new memory.

    format fields:
       fcn_suffix - 
    """
    fmt.hnameproto = wformat(
            "int {hnamefunc}\t(PyObject *obj,\t {PY_typedef_converter} *value)", fmt)
    fmt.__helper = CHelpers["get_from_object_char"]["name"]
    helper = dict(
        name=fmt.hnamefunc,
        dependent_helpers=[
            "PY_converter_type",
            "get_from_object_char",
        ],
        c_include=["<stdlib.h>"],   # malloc/free
        cxx_include=["<cstdlib>"],  # malloc/free
        proto=fmt.hnameproto + ";",
        source=wformat("""

// helper FREE_{hname}
static void FREE_{hname}(PyObject *obj)
{{+
char **in = {cast_static}char **{cast1}PyCapsule_GetPointer(obj, {nullptr}){cast2};
if (in == {nullptr})
+return;-
size_t *size = {cast_static}size_t *{cast1}PyCapsule_GetContext(obj){cast2};
if (size == {nullptr})
+return;-
for (size_t i=0; i < *size; ++i) {{+
if (in[i] == {nullptr})
+continue;-
{stdlib}free(in[i]);
-}}
{stdlib}free(in);
{stdlib}free(size);
-}}

// helper {hname}
// Convert obj into an array of char * (i.e. char **).
{PY_helper_static}{hnameproto}
{{+
PyObject *seq = PySequence_Fast(obj, "holder");
if (seq == NULL) {{+
PyErr_Format(PyExc_TypeError,\t "argument '%s' must be iterable",\t value->name);
return -1;
-}}
Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
char **in = {cast_static}char **{cast1}{stdlib}calloc(size, sizeof(char *)){cast2};
PyObject *dataobj = PyCapsule_New(in, {nullptr}, FREE_{hname});
size_t *size_context = {cast_static}size_t *{cast1}malloc(sizeof(size_t)){cast2};
*size_context = size;
int ierr = PyCapsule_SetContext(dataobj, size_context);
// XXX - check error
{PY_typedef_converter} itemvalue = {PY_value_init};
for (Py_ssize_t i = 0; i < size; i++) {{+
PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
ierr = {__helper}(item, &itemvalue);
if (ierr == 0) {{+
Py_XDECREF(itemvalue.dataobj);
Py_DECREF(dataobj);
Py_DECREF(seq);
PyErr_Format(PyExc_TypeError,\t "argument '%s', index %d must be {fcn_type}",\t value->name,\t (int) i);
return 0;
-}}
if (itemvalue.data != {nullptr}) {{+
in[i] = strdup({cast_static}char *{cast1}itemvalue.data{cast2});
-}}
Py_XDECREF(itemvalue.dataobj);
-}}
Py_DECREF(seq);

value->obj = {nullptr};
value->dataobj = dataobj;
value->data = in;
value->size = {size_var};
return 1;
-}}""", fmt),
    )
    return helper

def add_to_PyList_helper_vector(fmt, ntypemap):
    """Add helpers to work with Python lists.
    Several helpers are created based on the type of arg.
    Used with sgroup="native" types.

    Args:
        fmt      - util.Scope
        ntypemap - typemap.Typemap
    """
    flat_name = ntypemap.flat_name
    fmt.c_type = ntypemap.c_type
    fmt.cxx_type = ntypemap.cxx_type
    
    # Used with intent(out)
    name = "to_PyList_vector_" + flat_name
    ctor = ntypemap.PY_ctor
    if ctor is None:
        ctor = "XXXPy_ctor"
    ctor_expr = "in[i]"
    if ntypemap.py_ctype is not None:
        ctor_expr = ntypemap.pytype_to_pyctor.format(ctor_expr=ctor_expr)
    fmt.Py_ctor = ctor.format(ctor_expr=ctor_expr)
    fmt.hname = name
    fmt.hnamefunc = wformat("{PY_helper_prefix}{hname}", fmt)
    fmt.hnameproto = wformat("PyObject *{hnamefunc}\t(std::vector<{c_type}> & in)", fmt)
    helper = dict(
        name=fmt.hnamefunc,
        proto=fmt.hnameproto + ";",
        source=wformat(
            """
// helper {hname}
{PY_helper_static}{hnameproto}
{{+
size_t size = in.size();
PyObject *out = PyList_New(size);
for (size_t i = 0; i < size; ++i) {{+
PyList_SET_ITEM(out, i, {Py_ctor});
-}}
return out;
-}}""",
            fmt,
        ),
    )
    CHelpers[name] = helper

    # Used with intent(inout)
    name = "update_PyList_vector_" + flat_name
    ctor = ntypemap.PY_ctor
    if ctor is None:
        ctor = "XXXPy_ctor"
    ctor_expr = "in[i]"
    if ntypemap.py_ctype is not None:
        ctor_expr = ntypemap.pytype_to_pyctor.format(ctor_expr=ctor_expr)
    fmt.Py_ctor = ctor.format(ctor_expr=ctor_expr)
    fmt.hname = name
    fmt.hnamefunc = wformat(
        "{PY_helper_prefix}{hname}", fmt)
    fmt.hnameproto = wformat(
        "void {hnamefunc}\t(PyObject *out, {c_type} *in, size_t size)", fmt)
    helper = dict(
        name=fmt.hnamefunc,
        proto=fmt.hnameproto + ";",
        source=wformat(
            """
// helper {hname}
// Replace members of existing list with new values.
// out is known to be a PyList of the correct length.
{PY_helper_static}{hnameproto}
{{+
for (size_t i = 0; i < size; ++i) {{+
PyObject *item = PyList_GET_ITEM(out, i);
Py_DECREF(item);
PyList_SET_ITEM(out, i, {Py_ctor});
-}}
-}}""",
            fmt,
        ),
    )
    CHelpers[name] = helper

    # used with intent(in)
    # Return -1 on error.
    # Convert an empty list into a NULL pointer.
    # Use a fixed text in PySequence_Fast.
    # If an error occurs, replace message with one which includes argument name.
    name = "create_from_PyObject_vector_" + flat_name
    get = ntypemap.PY_get
    if get is None:
        get = "XXXPy_get"
    py_var = "item"
    fmt.Py_get = get.format(py_var=py_var)
    fmt.py_ctype = fmt.c_type;
    fmt.work_ctor = "cvalue"
    if ntypemap.py_ctype is not None:
        fmt.py_ctype = ntypemap.py_ctype
        fmt.work_ctor = ntypemap.pytype_to_cxx.format(work_var=fmt.work_ctor)
    fmt.hname = name
    fmt.hnamefunc= wformat(
        "{PY_helper_prefix}{hname}", fmt)
    fmt.hnameproto = wformat(
        "int {hnamefunc}\t(PyObject *obj,\t const char *name,\t std::vector<{cxx_type}> & in)", fmt)
    helper = dict(
        name=fmt.hnamefunc,
##-        cxx_include=["<cstdlib>"],  # malloc/free
        cxx_proto=fmt.hnameproto + ";",
        cxx_source=wformat(
            """
// helper {hname}
// Convert obj into an array of type {cxx_type}
// Return -1 on error.
{PY_helper_static}{hnameproto}
{{+
PyObject *seq = PySequence_Fast(obj, "holder");
if (seq == NULL) {{+
PyErr_Format(PyExc_TypeError,\t "argument '%s' must be iterable",\t name);
return -1;
-}}
Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
for (Py_ssize_t i = 0; i < size; i++) {{+
PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
{py_ctype} cvalue = {Py_get};
if (PyErr_Occurred()) {{+
Py_DECREF(seq);
PyErr_Format(PyExc_ValueError,\t "argument '%s', index %d must be {c_type}",\t name,\t (int) i);
return -1;
-}}
in.push_back({work_ctor});
-}}
Py_DECREF(seq);
return 0;
-}}""",
            fmt,
        ),
    )
    CHelpers[name] = helper

"""
http://effbot.org/zone/python-capi-sequences.htm
if (PyList_Check(seq))
        for (i = 0; i < len; i++) {
            item = PyList_GET_ITEM(seq, i);
            ...
        }
    else
        for (i = 0; i < len; i++) {
            item = PyTuple_GET_ITEM(seq, i);
            ...
        }
"""
    
######################################################################
# Static helpers

CHelpers = dict(
    type_defines=dict(
        # Order derived from TS 29113
        # with the addition of unsigned types
        scope="cwrap_include",
        source="""
/* helper type_defines */
/* Shroud type defines */
#define SH_TYPE_SIGNED_CHAR 1
#define SH_TYPE_SHORT       2
#define SH_TYPE_INT         3
#define SH_TYPE_LONG        4
#define SH_TYPE_LONG_LONG   5
#define SH_TYPE_SIZE_T      6

#define SH_TYPE_UNSIGNED_SHORT       SH_TYPE_SHORT + 100
#define SH_TYPE_UNSIGNED_INT         SH_TYPE_INT + 100
#define SH_TYPE_UNSIGNED_LONG        SH_TYPE_LONG + 100
#define SH_TYPE_UNSIGNED_LONG_LONG   SH_TYPE_LONG_LONG + 100

#define SH_TYPE_INT8_T      7
#define SH_TYPE_INT16_T     8
#define SH_TYPE_INT32_T     9
#define SH_TYPE_INT64_T    10

#define SH_TYPE_UINT8_T    SH_TYPE_INT8_T + 100
#define SH_TYPE_UINT16_T   SH_TYPE_INT16_T + 100
#define SH_TYPE_UINT32_T   SH_TYPE_INT32_T + 100
#define SH_TYPE_UINT64_T   SH_TYPE_INT64_T + 100

/* least8 least16 least32 least64 */
/* fast8 fast16 fast32 fast64 */
/* intmax_t intptr_t ptrdiff_t */

#define SH_TYPE_FLOAT        22
#define SH_TYPE_DOUBLE       23
#define SH_TYPE_LONG_DOUBLE  24
#define SH_TYPE_FLOAT_COMPLEX       25
#define SH_TYPE_DOUBLE_COMPLEX      26
#define SH_TYPE_LONG_DOUBLE_COMPLEX 27

#define SH_TYPE_BOOL       28
#define SH_TYPE_CHAR       29
#define SH_TYPE_CPTR       30
#define SH_TYPE_STRUCT     31
#define SH_TYPE_OTHER      32""",
    ),
    char_copy=dict(
        name="ShroudCharCopy",
        c_include=["<string.h>"],
        c_source="""
// helper ShroudCharCopy
// Copy src into dest, blank fill to ndest characters
// Truncate if dest is too short.
// dest will not be NULL terminated.
static void ShroudCharCopy(char *dest, int ndest, const char *src, int nsrc)
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
        cxx_include=["<cstring>"],
        cxx_source="""
// helper ShroudCharCopy
// Copy src into dest, blank fill to ndest characters
// Truncate if dest is too short.
// dest will not be NULL terminated.
static void ShroudCharCopy(char *dest, int ndest, const char *src, int nsrc)
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
    char_blank_fill=dict(
        name="ShroudCharBlankFill",
        c_include=["<string.h>"],
        c_source="""
// helper char_blank_fill
// blank fill dest starting at trailing NULL.
static void ShroudCharBlankFill(char *dest, int ndest)
{
   int nm = strlen(dest);
   if(ndest > nm) memset(dest+nm,' ',ndest-nm);
}""",
        cxx_include=["<cstring>"],
        cxx_source="""
// helper char_blank_fill
// blank fill dest starting at trailing NULL.
static void ShroudCharBlankFill(char *dest, int ndest)
{
   int nm = std::strlen(dest);
   if(ndest > nm) std::memset(dest+nm,' ',ndest-nm);
}""",
    ),

    ########################################
    # Used by 'const char *' arguments which need to be NULL terminated
    # in the C wrapper.
    char_alloc=dict(
        name="ShroudCharAlloc",
        c_include=["<string.h>", "<stdlib.h>", "<stddef.h>"],
        c_source="""
// helper char_alloc
// Copy src into new memory and null terminate.
// If ntrim is 0, return NULL pointer.
// If blanknull is 1, return NULL when string is blank.
static char *ShroudCharAlloc(const char *src, int nsrc, int blanknull)
{
   int ntrim = ShroudCharLenTrim(src, nsrc);
   if (ntrim == 0 && blanknull == 1) {
     return NULL;
   }
   char *rv = malloc(nsrc + 1);
   if (ntrim > 0) {
     memcpy(rv, src, ntrim);
   }
   rv[ntrim] = '\\0';
   return rv;
}""",
        cxx_include=["<cstring>", "<cstdlib>"],
        cxx_source="""
// helper char_alloc
// Copy src into new memory and null terminate.
// If ntrim is 0, return NULL pointer.
// If blanknull is 1, return NULL when string is blank.
static char *ShroudCharAlloc(const char *src, int nsrc, int blanknull)
{
   int ntrim = ShroudCharLenTrim(src, nsrc);
   if (ntrim == 0 && blanknull == 1) {
     return nullptr;
   }
   char *rv = (char *) std::malloc(nsrc + 1);
   if (ntrim > 0) {
     std::memcpy(rv, src, ntrim);
   }
   rv[ntrim] = '\\0';
   return rv;
}""",
        dependent_helpers=["char_len_trim"],
    ),

    char_free=dict(
        name="ShroudCharFree",
        c_include=["<stdlib.h>"],
        c_source="""
// helper char_free
// Release memory allocated by ShroudCharAlloc
static void ShroudCharFree(char *src)
{
   if (src != NULL) {
     free(src);
   }
}""",
        cxx_include=["<cstdlib>"],
        cxx_source="""
// helper char_free
// Release memory allocated by ShroudCharAlloc
static void ShroudCharFree(char *src)
{
   if (src != NULL) {
     std::free(src);
   }
}""",
    ),

    ########################################
    char_len_trim=dict(
        name="ShroudCharLenTrim",
        source="""
// helper char_len_trim
// Returns the length of character string src with length nsrc,
// ignoring any trailing blanks.
static int ShroudCharLenTrim(const char *src, int nsrc) {
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
    ########################################
    # Used with 'char **' arguments.
    char_array_alloc=dict(
        name="ShroudStrArrayAlloc",
        dependent_helpers=["char_len_trim"],
        c_include=["<string.h>", "<stdlib.h>"],
        c_source="""
// helper char_array_alloc
// Copy src into new memory and null terminate.
static char **ShroudStrArrayAlloc(const char *src, int nsrc, int len)
{
   char **rv = malloc(sizeof(char *) * nsrc);
   const char *src0 = src;
   for(int i=0; i < nsrc; ++i) {
      int ntrim = ShroudCharLenTrim(src0, len);
      char *tgt = malloc(ntrim+1);
      memcpy(tgt, src0, ntrim);
      tgt[ntrim] = '\\0';
      rv[i] = tgt;
      src0 += len;
   }
   return rv;
}""",
        cxx_include=["<cstring>", "<cstdlib>"],
        cxx_source="""
// helper char_array_alloc
// Copy src into new memory and null terminate.
// char **src +size(nsrc) +len(len)
// CHARACTER(len) src(nsrc)
static char **ShroudStrArrayAlloc(const char *src, int nsrc, int len)
{
   char **rv = static_cast\t<char **>\t(std::malloc(sizeof(char *) * nsrc));
   const char *src0 = src;
   for(int i=0; i < nsrc; ++i) {
      int ntrim = ShroudCharLenTrim(src0, len);
      char *tgt = static_cast<char *>(std::malloc(ntrim+1));
      std::memcpy(tgt, src0, ntrim);
      tgt[ntrim] = '\\0';
      rv[i] = tgt;
      src0 += len;
   }
   return rv;
}""",
    ),
    
    char_array_free=dict(
        name="ShroudStrArrayFree",
        c_include=["<stdlib.h>"],
        c_source="""
// helper char_array_free
// Release memory allocated by ShroudStrArrayAlloc
static void ShroudStrArrayFree(char **src, int nsrc)
{
   for(int i=0; i < nsrc; ++i) {
       free(src[i]);
   }
   free(src);
}""",
        cxx_include=["<cstdlib>"],
        cxx_source="""
// helper char_array_free
// Release memory allocated by ShroudStrArrayAlloc
static void ShroudStrArrayFree(char **src, int nsrc)
{
   for(int i=0; i < nsrc; ++i) {
       std::free(src[i]);
   }
   std::free(src);
}""",
    ),
    ########################################
    # Find size of CFI array
    size_CFI=dict(
        c_include=["<stddef.h>"],
        cxx_include=["<cstddef>"],
        source="""
// helper size_CFI
// Compute number of items in CFI_cdesc_t
size_t ShroudSizeCFI(CFI_cdesc_t *desc)
{
    size_t nitems = 1;
    for (int i = 0; i < desc->rank; i++) {
        nitems *= desc->dim[i].extent;
    }
    return nitems;
}""",
    ),
    ########################################
)   # end CHelpers


FHelpers = dict(
    type_defines=dict(
        derived_type="""
! helper type_defines
! Shroud type defines from helper type_defines
integer, parameter, private :: &
    SH_TYPE_SIGNED_CHAR= 1, &
    SH_TYPE_SHORT      = 2, &
    SH_TYPE_INT        = 3, &
    SH_TYPE_LONG       = 4, &
    SH_TYPE_LONG_LONG  = 5, &
    SH_TYPE_SIZE_T     = 6, &
    SH_TYPE_UNSIGNED_SHORT      = SH_TYPE_SHORT + 100, &
    SH_TYPE_UNSIGNED_INT        = SH_TYPE_INT + 100, &
    SH_TYPE_UNSIGNED_LONG       = SH_TYPE_LONG + 100, &
    SH_TYPE_UNSIGNED_LONG_LONG  = SH_TYPE_LONG_LONG + 100, &
    SH_TYPE_INT8_T    =  7, &
    SH_TYPE_INT16_T   =  8, &
    SH_TYPE_INT32_T   =  9, &
    SH_TYPE_INT64_T   = 10, &
    SH_TYPE_UINT8_T  =  SH_TYPE_INT8_T + 100, &
    SH_TYPE_UINT16_T =  SH_TYPE_INT16_T + 100, &
    SH_TYPE_UINT32_T =  SH_TYPE_INT32_T + 100, &
    SH_TYPE_UINT64_T =  SH_TYPE_INT64_T + 100, &
    SH_TYPE_FLOAT       = 22, &
    SH_TYPE_DOUBLE      = 23, &
    SH_TYPE_LONG_DOUBLE = 24, &
    SH_TYPE_FLOAT_COMPLEX      = 25, &
    SH_TYPE_DOUBLE_COMPLEX     = 26, &
    SH_TYPE_LONG_DOUBLE_COMPLEX= 27, &
    SH_TYPE_BOOL      = 28, &
    SH_TYPE_CHAR      = 29, &
    SH_TYPE_CPTR      = 30, &
    SH_TYPE_STRUCT    = 31, &
    SH_TYPE_OTHER     = 32""",
    ),
)  # end FHelpers



########################################
# Routines to dump helper routines to a file.

def gather_helpers(fp, wrapper, helpers, keys):
    """Dump helpers in human readable format.
    Dump selected keys in a format which can be used with sphinx
    literalinclude. Dump the other keys as JSON.
    Use with testing.
    """
    for name in sorted(helpers.keys()):
        helper = helpers[name]
        out = {}
        output = []
        for key, value in helper.items():
            if key in keys:
                output.append("")
                output.append("##### start {} {}".format(name, key))
                output.append(helper[key])
                output.append("##### end {} {}".format(name, key))
            else:
                out[key] = value

        print("\n----------", name, "----------", file=fp)
        json.dump(out, fp, sort_keys=True, indent=4, separators=(',', ': '))
        print("", file=fp)
        wrapper.write_lines(fp, output)

    return

def write_c_helpers(fp):
    wrapper = util.WrapperMixin()
    wrapper.linelen = 72
    wrapper.indent = 0
    wrapper.cont = ""
    output = gather_helpers(fp, wrapper, CHelpers, ["source", "c_source", "cxx_source"])

def write_f_helpers(fp):
    wrapper = util.WrapperMixin()
    wrapper.linelen = 72
    wrapper.indent = 0
    wrapper.cont = "&"
    output = gather_helpers(fp, wrapper, FHelpers, ["derived_type", "interface", "source"])


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
        COMMAND ${CMAKE_COMMAND} -E touch ${_timestamp}
        COMMENT "Running shroud ${arg_YAML_INPUT_FILE}"
        WORKING_DIRECTORY ${SHROUD_OUTPUT_DIR}
    )

    # Create target to process this Shroud file
    add_custom_target(generate_${_basename}    DEPENDS ${_timestamp})

    add_dependencies(generate generate_${_basename})
endmacro(add_shroud)
"""
