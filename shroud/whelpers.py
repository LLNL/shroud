# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
"""
Helper functions for C and Fortran wrappers.


 C helper functions which may be added to a implementation file.

 name        = Name of function created by the helper function.
               This allows the function name to be independent
               of the helper name so that it may include a prefix
               to help control namespace/scope.
               Useful when to helpers create the same function.
               ex. SHROUD_get_from_object_char_{numpy,list}
 scope       = scope of helper.  Defaults to "file" which are added
               as file static and may be in several files.
               "utility" will add to C_header_utility or PY_utility_filename
               and shared among files. These names need to be unique
               since they are shared across wrapped libraries.
 c_include   = Blank delimited list of files to #include
               in implementation file when wrapping a C library.
 cxx_include = Blank delimited list of files to #include.
               in implementation file when wrapping a C++ library.
 c_source    = language=c source.
 cxx_source  = language=c++ source.
 dependent_helpers = list of helpers names needed by this helper
                     They will be added to the output before current helper.
 need_numpy  = If True, NumPy headers will be added.

 proto       = prototype for function.
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



"""

from . import typemap
from . import util

wformat = util.wformat

# Used with literalinclude
# format fields {lstart} and {lend}
cstart = "// start "
cend   = "// end "
fstart = "! start "
fend   = "! end "

_literalinclude = False
def set_literalinclude(flag):
    global _literalinclude
    _literalinclude = flag

_newlibrary = None
def set_library(library):
    global _newlibrary
    _newlibrary = library

def XXXwrite_helper_files(self, directory):
    """This library file is no longer needed.

    Should be written to config.c_fortran_dir
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


def add_external_helpers(fmtin, literalinclude):
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
    fmt = util.Scope(fmtin)
    fmt.lstart = ""
    fmt.lend = ""

    # Only used with std::string and thus C++
    name = "copy_string"
    fmt.hname = name
    if literalinclude:
        fmt.lstart = "{}helper {}\n".format(cstart, name)
        fmt.lend = "\n{}helper {}".format(cend, name)
    CHelpers[name] = dict(
        dependent_helpers=["array_context"],
        cxx_include="<cstring> <cstddef>",
        # XXX - mangle name
        source=wformat(
            """
// helper {hname}
{lstart}// Copy the char* or std::string in context into c_var.
// Called by Fortran to deal with allocatable character.
void {C_prefix}ShroudCopyStringAndFree({C_array_type} *data, char *c_var, size_t c_var_len) {{+
const char *cxx_var = data->addr.ccharp;
size_t n = c_var_len;
if (data->elem_len < n) n = data->elem_len;
{stdlib}strncpy(c_var, cxx_var, n);
{C_memory_dtor_function}(&data->cxx); // delete data->cxx.addr
-}}{lend}
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
! helper {hname}
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

    name = "ShroudStrToArray"
    fmt.hname = name
    if literalinclude:
        fmt.lstart = "{}helper {}\n".format(cstart, name)
        fmt.lend = "\n{}helper {}".format(cend, name)
    CHelpers[name] = dict(
        dependent_helpers=["array_context"],
        cxx_include="<cstring> <cstddef>",
        source=wformat(
            """
// helper {hname}
{lstart}// Save str metadata into array to allow Fortran to access values.
static void ShroudStrToArray({C_array_type} *array, const std::string * src, int idtor)
{{+
array->cxx.addr = static_cast<void *>(const_cast<std::string *>(src));
array->cxx.idtor = idtor;
if (src->empty()) {{+
array->addr.ccharp = NULL;
array->elem_len = 0;
-}} else {{+
array->addr.ccharp = src->data();
array->elem_len = src->length();
-}}
array->size = 1;
array->rank = 1;
-}}{lend}""", fmt),
    )
    ##########
    # Generate C or C++ version of helper.
    ##########
    # 'char *' needs a custom handler because of the nature
    # of NULL terminated strings.
    ntypemap = typemap.lookup_type("char")
    fmt.fcn_suffix = "char"
    fmt.fcn_type = "string"
    fmt.c_type = "char *"
    fmt.Py_get = "PyString_AsString(item)"
#    fmt.Py_get = ntypemap.PY_get.format(py_var="item")
    fmt.Py_ctor = ntypemap.PY_ctor.format(c_deref="", c_var="in[i]")
    fmt.hname = "from_PyObject_char"
    CHelpers["from_PyObject_char"] = create_from_PyObject(fmt)
    fmt.hname = "to_PyList_char"
    CHelpers["to_PyList_char"] = create_to_PyList(fmt)

    ########################################
    # char **
    name = "get_from_object_charptr"
    fmt.size_var="size"
    fmt.c_var="in"
    fmt.hname = name
    fmt.hnamefunc = fmt.PY_helper_prefix + name
    CHelpers[name] = create_get_from_object_list(fmt)
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
// Converter to PyObject to char *.
{PY_helper_static}{hnameproto}
{{+
char *out;
if (PyUnicode_Check(obj)) {{+
^#if PY_MAJOR_VERSION >= 3
PyObject *strobj = PyUnicode_AsUTF8String(obj);
out = PyBytes_AS_STRING(strobj);
value->obj = strobj;  // steal reference
^#else
PyObject *strobj = PyUnicode_AsUTF8String(obj);
out = PyString_AsString(strobj);
value->obj = strobj;  // steal reference
^#endif
^#if PY_MAJOR_VERSION >= 3
-}} else if (PyByteArray_Check(obj)) {{+
out = PyBytes_AS_STRING(obj);
value->obj = obj;
Py_INCREF(obj);
^#else
-}} else if (PyString_Check(obj)) {{+
out = PyString_AsString(obj);
value->obj = obj;
Py_INCREF(obj);
^#endif
-}} else if (obj == Py_None) {{+
out = NULL;
value->obj = NULL;
-}} else {{+
PyErr_SetString(PyExc_ValueError, "argument must be a string");
return 0;
-}}
value->data = out;
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

    ########################################
    CHelpers['PY_converter_type'] = dict(
        scope="utility",
        source=wformat("""
// helper PY_converter_type
// Store PyObject and pointer to the data it contains.
typedef struct {{+
PyObject *obj;
void *data;   // points into obj.
-}} {PY_typedef_converter};""", fmt)
    )
    
######################################################################

def add_shadow_helper(node):
    """
    Add helper functions for each shadow type.

    Args:
        node -
    """
    cname = node.typemap.c_type

    name = "capsule_{}".format(cname)
    if name not in CHelpers:
        if node.options.literalinclude:
            lstart = "{}struct {}\n".format(cstart, cname)
            lend = "\n{}struct {}".format(cend, cname)
        else:
            lstart = ""
            lend = ""
        if node.cpp_if:
            cpp_if = "#" + node.cpp_if + "\n"
            cpp_endif = "\n#endif  // " + node.cpp_if
        else:
            cpp_if = ""
            cpp_endif = ""
        helper = dict(
            scope="utility",
            # h_shared_code
            source="""
// helper {hname}
{lstart}{cpp_if}struct s_{C_type_name} {{+
void *addr;     /* address of C++ memory */
int idtor;      /* index of destructor */
-}};
typedef struct s_{C_type_name} {C_type_name};{cpp_endif}{lend}""".format(
                hname=name, C_type_name=cname,
                cpp_if=cpp_if, cpp_endif=cpp_endif,
                lstart=lstart, lend=lend,
            )
        )
        CHelpers[name] = helper
    return name


def add_capsule_helper(fmtin, literalinclude):
    """Share info with C++ to allow Fortran to release memory.

    Used with shadow classes and std::vector.

    Args:
        fmt - format dictionary from the library.
        literalinclude -
    """
    # Add some format strings
    fmt = util.Scope(fmtin)
    fmt.lstart = ""
    fmt.lend = ""

    name = "capsule_data_helper"
    fmt.hname = name
    if name not in FHelpers:
        helper = dict(
            derived_type=wformat(
                """
! helper {hname}
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
            scope="utility",
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
    if name not in FHelpers:
        # XXX split helper into to parts, one for each derived type
        helper = dict(
            dependent_helpers=["capsule_data_helper"],
            derived_type=wformat(
                """
! helper {hname}
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
! helper {hname}
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
    fmt.hname = name
    if name not in CHelpers:
        if literalinclude:
            fmt.lstart = "{}{}\n".format(cstart, name)
            fmt.lend = "\n{}{}".format(cend, name)
        helper = dict(
            scope="utility",
            include="<stddef.h>",
            # Create a union for addr to avoid some casts.
            # And help with debugging since ccharp will display contents.
            source=wformat(
                """
// helper {hname}
{lstart}struct s_{C_array_type} {{+
{C_capsule_data_type} cxx;      /* address of C++ memory */
union {{+
const void * base;
const char * ccharp;
-}} addr;
int type;        /* type of element */
size_t elem_len; /* bytes-per-item or character len in c++ */
size_t size;     /* size of data in c++ */
int rank;        /* number of dimensions */
-}};
typedef struct s_{C_array_type} {C_array_type};{lend}""",
                fmt,
            ),
            dependent_helpers=["capsule_data_helper"],
        )
        CHelpers[name] = helper

    if name not in FHelpers:
        # Create a derived type used to communicate with C wrapper.
        # Should never be exposed to user.
        if literalinclude:
            fmt.lstart = "{}{}\n".format(fstart, name)
            fmt.lend = "\n{}{}".format(fend, name)
        helper = dict(
            derived_type=wformat(
                """
! helper {hname}
{lstart}type, bind(C) :: {F_array_type}+
! address of C++ memory
type({F_capsule_data_type}) :: cxx
! address of data in cxx
type(C_PTR) :: base_addr = C_NULL_PTR
! type of element
integer(C_INT) :: type
! bytes-per-item or character len of data in cxx
integer(C_SIZE_T) :: elem_len = 0_C_SIZE_T
! size of data in cxx
integer(C_SIZE_T) :: size = 0_C_SIZE_T
! number of dimensions
integer(C_INT) :: rank = -1
-end type {F_array_type}{lend}""",
                fmt,
            ),
            modules=dict(iso_c_binding=["C_NULL_PTR", "C_PTR", "C_SIZE_T"]),
            dependent_helpers=["capsule_data_helper"],
        )
        FHelpers[name] = helper


def add_copy_array_helper_c(fmtin):
    """Create function to copy contents of a vector.

    The function has C_prefix in the name since it is not file static.
    This allows multiple wrapped libraries to coexist.

    Args:
        fmtin - format dictionary from the library.
    """
    fmt = util.Scope(fmtin)
    fmt.lstart = ""
    fmt.lend = ""

    name = "copy_array"
    fmt.hname = name
    if _literalinclude:
        fmt.lstart = "{}helper {}\n".format(cstart, name)
        fmt.lend = "\n{}helper {}".format(cend, name)
    if name not in CHelpers:
        helper = dict(
            dependent_helpers=["array_context"],
            c_include="<string.h>",
            cxx_include="<cstring>",
            # Create a single C routine which is called from Fortran
            # via an interface for each cxx_type.
            cxx_source=wformat(
                """
// helper {hname}
{lstart}// Copy std::vector into array c_var(c_var_size).
// Then release std::vector.
// Called from Fortran.
void {C_prefix}ShroudCopyArray({C_array_type} *data, \tvoid *c_var, \tsize_t c_var_size)
{{+
const void *cxx_var = data->addr.base;
int n = c_var_size < data->size ? c_var_size : data->size;
n *= data->elem_len;
{stdlib}memcpy(c_var, cxx_var, n);
{C_memory_dtor_function}(&data->cxx); // delete data->cxx.addr
-}}{lend}""",
                fmt,
            ),
        )
        CHelpers[name] = helper


def add_copy_array_helper(fmtin, ast):
    """Create Fortran interface to helper function
    which copies an array based on c_type.
    Each interface calls the same C helper.

    The function has C_prefix in the name since it is not file static.
    This allows multiple wrapped libraries to coexist.

    Args:
        fmtin -
        ast -
    """
    fmt = util.Scope(fmtin)
    ntypemap = ast.typemap
    if ntypemap.base == "vector":
        ntypemap = ast.template_arguments[0].typemap

    fmt.c_type = ntypemap.c_type
    fmt.f_kind = ntypemap.f_kind
    fmt.f_type = ntypemap.f_type

    name = wformat("copy_array_{c_type}", fmt)
    fmt.hname = name
    if name not in FHelpers:
        helper = dict(
            # XXX when f_kind == C_SIZE_T
            dependent_helpers=["array_context"],
            interface=wformat(
                """
interface+
! helper {hname}
! Copy contents of context into c_var.
subroutine SHROUD_copy_array_{c_type}(context, c_var, c_var_size) &+
bind(C, name="{C_prefix}ShroudCopyArray")
use iso_c_binding, only : {f_kind}, C_SIZE_T
import {F_array_type}
type({F_array_type}), intent(IN) :: context
{f_type}, intent(OUT) :: c_var(*)
integer(C_SIZE_T), value :: c_var_size
-end subroutine SHROUD_copy_array_{c_type}
-end interface""",
                fmt,
            ),
        )
        FHelpers[name] = helper
    return name

def add_to_PyList_helper(arg):
    """Add helpers to work with Python lists.
    Several helpers are created based on the type of arg.

    Args:
        arg - declast.Declaration
    """
    ntypemap = arg.typemap
    if ntypemap.base == "vector":
        add_to_PyList_helper_vector(arg)
        return

    fmt = util.Scope(_newlibrary.fmtdict)
    fmt.c_type = ntypemap.c_type
    
    ########################################
    # Used with intent(out)
    name = "to_PyList_" + ntypemap.c_type
    if name not in CHelpers:
        fmt.hname = name
        fmt.fcn_suffix = ntypemap.c_type
        fmt.Py_ctor = ntypemap.PY_ctor.format(c_deref="", c_var="in[i]")
        helper = create_to_PyList(fmt)
        CHelpers[name] = create_to_PyList(fmt)

    ########################################
    # Used with intent(inout)
    name = "update_PyList_" + ntypemap.c_type
    if name not in CHelpers:
        fmt.Py_ctor = ntypemap.PY_ctor.format(c_deref="", c_var="in[i]")
        fmt.hname = name
        fmt.hnameproto = wformat(
            "void {PY_helper_prefix}update_PyList_{c_type}\t(PyObject *out, {c_type} *in, size_t size)", fmt)
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
    # used with intent(in)
    # Return -1 on error.
    # Convert an empty list into a NULL pointer.
    # Use a fixed text in PySequence_Fast.
    # If an error occurs, replace message with one which includes argument name.
    name = "from_PyObject_" + ntypemap.c_type
    if name not in CHelpers and ntypemap.PY_get:
        fmt.hname = name
        fmt.fcn_suffix = ntypemap.c_type
        fmt.fcn_type = ntypemap.c_type
        fmt.Py_get = ntypemap.PY_get.format(py_var="item")
        CHelpers[name] = create_from_PyObject(fmt)

    ########################################
    # Function called by typemap.PY_get_converter for NumPy.
    name = "get_from_object_{}_numpy".format(ntypemap.c_type)
    if name not in CHelpers:
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
value->data = PyArray_DATA({cast_reinterpret}PyArrayObject *{cast1}{py_tmp}{cast2});
return 1;
-}}""", fmt),
        )
        CHelpers[name] = helper

    ########################################
    # Function called by typemap.PY_get_converter for list.
    name = "get_from_object_{}_list".format(ntypemap.c_type)
    if name not in CHelpers:
        fmt.size_var = "size"
        fmt.c_var = "in"
        fmt.fcn_suffix = ntypemap.c_type
        fmt.hname = name
        fmt.hnamefunc = fmt.PY_helper_prefix + name
        CHelpers[name] = create_get_from_object_list(fmt)

def create_from_PyObject(fmt):
    """Create helper to convert list of PyObjects to C array.
    """
    fmt.hnamefunc = wformat(
        "{PY_helper_prefix}from_PyObject_{fcn_suffix}", fmt)
    fmt.hnameproto = wformat(
            "int {hnamefunc}\t(PyObject *obj,\t const char *name,\t {c_type} **pin,\t Py_ssize_t *psize)", fmt)
    helper = dict(
        name=fmt.hnamefunc,
        c_include="<stdlib.h>",   # malloc/free
        cxx_include="<cstdlib>",  # malloc/free
        proto=fmt.hnameproto + ";",
        source=wformat(
                """
// helper {hname}
// Convert obj into an array of type {c_type}
// Return -1 on error.
{PY_helper_static}{hnameproto}
{{+
PyObject *seq = PySequence_Fast(obj, "holder");
if (seq == NULL) {{+
PyErr_Format(PyExc_TypeError,\t "argument '%s' must be iterable",\t name);
return -1;
-}}
Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
{c_type} *in = {cast_static}{c_type} *{cast1}{stdlib}malloc(size * sizeof({c_type})){cast2};
for (Py_ssize_t i = 0; i < size; i++) {{+
PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
in[i] = {Py_get};
if (PyErr_Occurred()) {{+
{stdlib}free(in);
Py_DECREF(seq);
PyErr_Format(PyExc_ValueError,\t "argument '%s', index %d must be {fcn_type}",\t name,\t (int) i);
return -1;
-}}
-}}
Py_DECREF(seq);
*pin = in;
*psize = size;
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
        "PyObject *{hnamefunc}\t({c_type} *in, size_t size)", fmt)
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

    format fields:
       fcn_suffix - 
    """
    fmt.hnameproto = wformat(
            "int {hnamefunc}\t(PyObject *obj,\t {PY_typedef_converter} *value)", fmt)
    helper = dict(
        name=fmt.hnamefunc,
        dependent_helpers=[
            "PY_converter_type",
            "from_PyObject_" + fmt.fcn_suffix,  #ntypemap.c_type,
        ],
        proto=fmt.hnameproto + ";",
        source=wformat("""
// helper {hname}
// Convert PyObject to {c_type} pointer.
{PY_helper_static}{hnameproto}
{{+
{c_type} *{c_var};
Py_ssize_t {size_var};
if ({PY_helper_prefix}from_PyObject_{fcn_suffix}\t(obj,\t \"{c_var}\",\t &{c_var}, \t &{size_var}) == -1) {{+
return 0;
-}}
value->obj = {nullptr};
value->data = {cast_static}{c_type} *{cast1}{c_var}{cast2};
return 1;
-}}""", fmt),
    )
    return helper

def add_to_PyList_helper_vector(arg):
    """Add helpers to work with Python lists.
    Several helpers are created based on the type of arg.

    Args:
        arg -
    """
    ntypemap = arg.typemap
    if ntypemap.base == "vector":
        ntypemap = arg.template_arguments[0].typemap

    fmt = util.Scope(_newlibrary.fmtdict)
    fmt.c_type = ntypemap.c_type
    
    # Used with intent(out)
    name = "to_PyList_vector_" + ntypemap.c_type
    if name not in CHelpers:
        fmt.Py_ctor = ntypemap.PY_ctor.format(c_deref="", c_var="in[i]")
        fmt.hname = name
        fmt.hnamefunc = wformat("{PY_helper_prefix}to_PyList_vector_{c_type}", fmt)
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
    name = "update_PyList_vector_" + ntypemap.c_type
    if name not in CHelpers:
        fmt.Py_ctor = ntypemap.PY_ctor.format(c_deref="", c_var="in[i]")
        fmt.hname = name
        fmt.hnamefunc = wformat(
            "{PY_helper_prefix}update_PyList_{c_type}", fmt)
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
    name = "from_PyObject_vector_" + ntypemap.c_type
    if name not in CHelpers:
        fmt.Py_get = ntypemap.PY_get.format(py_var="item")
        fmt.hname = name
        fmt.hnamefunc= wformat(
            "{PY_helper_prefix}from_PyObject_vector_{c_type}", fmt)
        fmt.hnameproto = wformat(
            "int {hnamefunc}\t(PyObject *obj,\t const char *name,\t std::vector<{c_type}> & in)", fmt)
        helper = dict(
            name=fmt.hnamefunc,
##-            cxx_include="<cstdlib>",  # malloc/free
            cxx_proto=fmt.hnameproto + ";",
            cxx_source=wformat(
                """
// helper {hname}
// Convert obj into an array of type {c_type}
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
in.push_back({Py_get});
if (PyErr_Occurred()) {{+
Py_DECREF(seq);
PyErr_Format(PyExc_ValueError,\t "argument '%s', index %d must be {c_type}",\t name,\t (int) i);
return -1;
-}}
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
    ShroudTypeDefines=dict(
        # Order derived from TS 29113
        # with the addition of unsigned types
        scope="utility",
        source="""
/* helper ShroudTypeDefines */
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
    ShroudStrCopy=dict(
        c_include="<string.h>",
        c_source="""
// helper ShroudStrCopy
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
        cxx_include="<cstring>",
        cxx_source="""
// helper ShroudStrCopy
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
    ShroudStrBlankFill=dict(
        c_include="<string.h>",
        c_source="""
// helper ShroudStrBlankFill
// blank fill dest starting at trailing NULL.
static void ShroudStrBlankFill(char *dest, int ndest)
{
   int nm = strlen(dest);
   if(ndest > nm) memset(dest+nm,' ',ndest-nm);
}""",
        cxx_include="<cstring>",
        cxx_source="""
// helper ShroudStrBlankFill
// blank fill dest starting at trailing NULL.
static void ShroudStrBlankFill(char *dest, int ndest)
{
   int nm = std::strlen(dest);
   if(ndest > nm) std::memset(dest+nm,' ',ndest-nm);
}""",
    ),

    ########################################
    # Used by 'const char *' arguments which need to be NULL terminated
    # in the C wrapper.
    ShroudStrAlloc=dict(
        c_include="<string.h> <stdlib.h>",
        c_source="""
// helper ShroudStrAlloc
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
        cxx_include="<cstring> <cstdlib>",
        cxx_source="""
// helper ShroudStrAlloc
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
        c_include="<stdlib.h>",
        c_source="""
// helper ShroudStrFree
// Release memory allocated by ShroudStrAlloc
static void ShroudStrFree(char *src)
{
   free(src);
}""",
        cxx_include="<cstdlib>",
        cxx_source="""
// helper ShroudStrFree
// Release memory allocated by ShroudStrAlloc
static void ShroudStrFree(char *src)
{
   free(src);
}""",
    ),

    ########################################
    ShroudLenTrim=dict(
        source="""
// helper ShroudLenTrim
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
    ########################################
    # Used with 'char **' arguments.
    ShroudStrArrayAlloc=dict(
        dependent_helpers=["ShroudLenTrim"],
        c_include="<string.h> <stdlib.h>",
        c_source="""
// helper ShroudStrArrayAlloc
// Copy src into new memory and null terminate.
static char **ShroudStrArrayAlloc(const char *src, int nsrc, int len)
{
   char **rv = malloc(sizeof(char *) * nsrc);
   const char *src0 = src;
   for(int i=0; i < nsrc; ++i) {
      int ntrim = ShroudLenTrim(src0, len);
      char *tgt = malloc(ntrim+1);
      memcpy(tgt, src0, ntrim);
      tgt[ntrim] = '\\0';
      rv[i] = tgt;
      src0 += len;
   }
   return rv;
}""",
        cxx_include="<cstring> <cstdlib>",
        cxx_source="""
// helper ShroudStrArrayAlloc
// Copy src into new memory and null terminate.
// char **src +size(nsrc) +len(len)
// CHARACTER(len) src(nsrc)
static char **ShroudStrArrayAlloc(const char *src, int nsrc, int len)
{
   char **rv = static_cast\t<char **>\t(std::malloc(sizeof(char *) * nsrc));
   const char *src0 = src;
   for(int i=0; i < nsrc; ++i) {
      int ntrim = ShroudLenTrim(src0, len);
      char *tgt = static_cast<char *>(std::malloc(ntrim+1));
      std::memcpy(tgt, src0, ntrim);
      tgt[ntrim] = '\\0';
      rv[i] = tgt;
      src0 += len;
   }
   return rv;
}""",
    ),
    
    ShroudStrArrayFree=dict(
        c_include="<stdlib.h>",
        c_source="""
// helper ShroudStrArrayFree
// Release memory allocated by ShroudStrArrayAlloc
static void ShroudStrArrayFree(char **src, int nsrc)
{
   for(int i=0; i < nsrc; ++i) {
       free(src[i]);
   }
   free(src);
}""",
        cxx_include="<cstdlib>",
        cxx_source="""
// helper ShroudStrArrayFree
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
)   # end CHelpers


FHelpers = dict(
    ShroudTypeDefines=dict(
        derived_type="""
! helper ShroudTypeDefines
! Shroud type defines from helper ShroudTypeDefines
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
