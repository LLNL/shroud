# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Create and manage typemaps used to convert between languages.
"""

from . import util
from . import whelpers

# translation table to convert type name to flat name
# unsigned int -> unsigned_int
# vector<int>  -> vector_int
try:
    # Python 2
    from string import maketrans
    def flatten_name(name, flat_trantab = maketrans("< ", "__")):
        return name.replace("::","_").translate(flat_trantab, ">")
except:
    def flatten_name(name, flat_trantab="".maketrans("< ", "__", ">")):
        return name.replace("::","_").translate(flat_trantab)

class Typemap(object):
    """Collect fields for an argument.
    This used to be a dict but a class has better access semantics:
       i.attr vs d['attr']
    It also initializes default values to avoid  d.get('attr', default)

    c_header and cxx_header are used for interface. For example,
    size_t uses <stddef.h> and <cstddef>.

    impl_header is used for implementation.  For example, std::string
    uses <string>. <string> should not be in the interface since the
    wrapper is a C API.

    """

    # Array of known keys with default values
    _order = (
        ("flat_name", None),  # Name when used by wrapper identifiers
        ("template_suffix", None),  # Name when used by wrapper identifiers
                                    # when added to class/struct format.
        ("base", "unknown"),  # Base type: 'string'
        ("forward", None),  # Forward declaration
        ("format", {}),  # Applied to Scope for variable.
        ("typedef", None),  # Initialize from existing type
        ("cpp_if", None),  # C preprocessor test for c_header
        ("idtor", "0"),  # index of capsule_data destructor
        ("cxx_type", None),  # Name of type in C++, including namespace
        ("cxx_to_c", None),  # Expression to convert from C++ to C
        # None implies {cxx_var} i.e. no conversion
        (
            "cxx_header",
            None,
        ),  # Name of C++ header file required for implementation
        # For example, if cxx_to_c was a function
        ("c_type", None),  # Name of type in C
        ("c_header", None),  # Name of C header file required for type
        ("c_to_cxx", None),  # Expression to convert from C to C++
        # None implies {c_var}  i.e. no conversion
        ("c_statements", {}),
        ("c_templates", {}),  # c_statements for cxx_T
        ("c_return_code", None),
        (
            "c_union",
            None,
        ),  # Union of C++ and C type (used with structs and complex)
        (
            "f_c_module",
            None,
        ),  # Fortran modules needed for interface  (dictionary)
        ("f_type", None),  # Name of type in Fortran -- integer(C_INT)
        ("f_kind", None),  # Fortran kind            -- C_INT
        ("f_c_type", None),  # Type for C interface    -- int
        ("f_to_c", None),  # Expression to convert from Fortran to C
        (
            "f_module_name",
            None,
        ),  # Name of module which contains f_derived_type and f_capsule_data_type
        ("f_derived_type", None),  # Fortran derived type name
        ("f_capsule_data_type", None),  # Fortran derived type to match C struct
        ("f_args", None),  # Argument in Fortran wrapper to call C.
        ("f_module", None),  # Fortran modules needed for type  (dictionary)
        ("f_cast", "{f_var}"),  # Expression to convert to type
        ("f_cast_module", None),  # Fortran modules needed for f_cast
        ("f_cast_keywords", None),  # Dictionary of additional arguments to gen_arg_as_fortran
                                     # dict(is_target=True)
        # e.g. intrinsics such as int and real
        ("f_statements", {}),
        # override fields when result should be treated as an argument
        ("result_as_arg", None),
        ("impl_header", None), # implementation header
        # Python
        ("PY_format", "O"),  # 'format unit' for PyArg_Parse
        ("PY_PyTypeObject", None),  # variable name of PyTypeObject instance
        ("PY_PyObject", None),  # typedef name of PyObject instance
        ("PY_ctor", None),  # expression to create object.
        # ex. PyBool_FromLong({rv})
        ("PY_get", None),  # expression to create type from PyObject.
        ("PY_to_object", None),  # PyBuild - object'=converter(address)
        (
            "PY_from_object",
            None,
        ),  # PyArg_Parse - status=converter(object, address);
        ("PY_build_arg", None),  # argument for Py_BuildValue
        ("PY_build_format", None),  # 'format unit' for Py_BuildValue
        ("PYN_typenum", None),  # NumPy typenum enumeration
        (
            "PYN_descr",
            None,
        ),  # Name of PyArray_Descr variable to describe type (for structs)
        ("py_statements", {}),
        # Lua
        ("LUA_type", "LUA_TNONE"),
        ("LUA_pop", "POP"),
        ("LUA_push", "PUSH"),
        ("LUA_statements", {}),
        ("__line__", None),
    )

    _keyorder, _valueorder = zip(*_order)

    # valid fields
    defaults = dict(_order)

    def __init__(self, name, **kw):
        """
        Args:
            name - name of typemap.
        """
        self.name = name
        #        for key, defvalue in self.defaults.items():
        #            setattr(self, key, defvalue)
        self.__dict__.update(self.defaults)  # set all default values
        self.update(kw)
        if self.cxx_type and not self.flat_name:
            # Do not override an explicitly set value.
            self.compute_flat_name()

    def update(self, dct):
        """Add options from dictionary to self.

        Args:
            dct - dictionary-like object.
        """
        for key in dct:
            if key in self.defaults:
                setattr(self, key, dct[key])
            else:
                raise RuntimeError("Unknown key for Typemap %s", key)

    def XXXcopy(self):
        ntypemap = Typemap(self.name)
        ntypemap.update(self._to_dict())
        return ntypemap

    def clone_as(self, name):
        """
        Args:
            name - name of new instance.
        """
        ntypemap = Typemap(name, **self._to_dict())
        return ntypemap

    def compute_flat_name(self):
        """Compute the flat_name.
        Can be called after cxx_type is set
        such as after clone_as.

        cxx_type will not be set for template arguments.
        """
        self.flat_name = flatten_name(self.cxx_type)

    def _to_dict(self):
        """Convert instance to a dictionary for json.
        """
        # only export non-default values
        dct = {}
        for key, defvalue in self.defaults.items():
            value = getattr(self, key)
            if value is not defvalue:
                dct[key] = value
        return dct

    def __repr__(self):
        # only print non-default values
        args = []
        for key, defvalue in self.defaults.items():
            value = getattr(self, key)
            if value is not defvalue:
                if isinstance(value, str):
                    args.append("{0}='{1}'".format(key, value))
                else:
                    args.append("{0}={1}".format(key, value))
        return "Typemap('%s', " % self.name + ",".join(args) + ")"

    def __as_yaml__(self, indent, output):
        """Write out entire typemap as YAML.

        Args:
            indent -
            output -
        """
        util.as_yaml(self, self._keyorder, indent, output)

    def __export_yaml__(self, indent, output):
        """Write out a subset of a wrapped type.
        Other fields are set with fill_shadow_typemap_defaults.

        Args:
            indent -
            output -
        """
        util.as_yaml(
            self,
            [
                "base",
                "impl_header",
                "cxx_header",
                "cxx_type",
                "c_type",
                "c_header",
                "f_module_name",
                "f_derived_type",
                "f_capsule_data_type",
                "f_to_c",
            ],
            indent,
            output,
        )


# Manage collection of typemaps
shared_typedict = {}  # dictionary of registered types


def set_global_types(typedict):
    global shared_typedict
    shared_typedict = typedict


def get_global_types():
    return shared_typedict


def register_type(name, intypemap):
    """Register a typemap"""
    shared_typedict[name] = intypemap


def lookup_type(name):
    """Lookup name in registered types."""
    outtypemap = shared_typedict.get(name)
    return outtypemap


def initialize():
    set_global_types({})
    def_types = dict(
        void=Typemap(
            "void",
            c_type="void",
            cxx_type="void",
            # fortran='subroutine',
            f_type="type(C_PTR)",
            f_module=dict(iso_c_binding=["C_PTR"]),
            f_cast="C_LOC({f_var})",    # Cast an argument to a void *.
            f_cast_module=dict(iso_c_binding=["C_LOC"]),
            f_cast_keywords=dict(is_target=True),
            PY_ctor="PyCapsule_New({cxx_var}, NULL, NULL)",
        ),
        short=Typemap(
            "short",
            c_type="short",
            cxx_type="short",
            f_cast="int({f_var}, C_SHORT)",
            f_type="integer(C_SHORT)",
            f_kind="C_SHORT",
            f_module=dict(iso_c_binding=["C_SHORT"]),
            PY_format="h",
            PY_ctor="PyInt_FromLong({c_deref}{c_var})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_SHORT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        int=Typemap(
            "int",
            c_type="int",
            cxx_type="int",
            f_cast="int({f_var}, C_INT)",
            f_type="integer(C_INT)",
            f_kind="C_INT",
            f_module=dict(iso_c_binding=["C_INT"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({c_deref}{c_var})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        long=Typemap(
            "long",
            c_type="long",
            cxx_type="long",
            f_cast="int({f_var}, C_LONG)",
            f_type="integer(C_LONG)",
            f_kind="C_LONG",
            f_module=dict(iso_c_binding=["C_LONG"]),
            PY_format="l",
            PY_ctor="PyInt_FromLong({c_deref}{c_var})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_LONG",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        long_long=Typemap(
            "long_long",
            c_type="long long",
            cxx_type="long long",
            f_cast="int({f_var}, C_LONG_LONG)",
            f_type="integer(C_LONG_LONG)",
            f_kind="C_LONG_LONG",
            f_module=dict(iso_c_binding=["C_LONG_LONG"]),
            PY_format="L",
            # #- PY_ctor='PyInt_FromLong({c_deref}{c_var})',
            PYN_typenum="NPY_LONGLONG",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        unsigned_short=Typemap(
            "unsigned_short",
            c_type="unsigned short",
            cxx_type="unsigned short",
            f_cast="int({f_var}, C_SHORT)",
            f_type="integer(C_SHORT)",
            f_kind="C_SHORT",
            f_module=dict(iso_c_binding=["C_SHORT"]),
            PY_format="h",
            PY_ctor="PyInt_FromLong({c_deref}{c_var})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_SHORT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        unsigned_int=Typemap(
            "unsigned_int",
            c_type="unsigned int",
            cxx_type="unsigned int",
            f_cast="int({f_var}, C_INT)",
            f_type="integer(C_INT)",
            f_kind="C_INT",
            f_module=dict(iso_c_binding=["C_INT"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({c_deref}{c_var})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        unsigned_long=Typemap(
            "unsigned_long",
            c_type="unsigned long",
            cxx_type="unsigned long",
            f_cast="int({f_var}, C_LONG)",
            f_type="integer(C_LONG)",
            f_kind="C_LONG",
            f_module=dict(iso_c_binding=["C_LONG"]),
            PY_format="l",
            PY_ctor="PyInt_FromLong({c_deref}{c_var})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_LONG",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        unsigned_long_long=Typemap(
            "unsigned_long_long",
            c_type="unsigned long long",
            cxx_type="unsigned long long",
            f_cast="int({f_var}, C_LONG_LONG)",
            f_type="integer(C_LONG_LONG)",
            f_kind="C_LONG_LONG",
            f_module=dict(iso_c_binding=["C_LONG_LONG"]),
            PY_format="L",
            # #- PY_ctor='PyInt_FromLong({c_deref}{c_var})',
            PYN_typenum="NPY_LONGLONG",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        size_t=Typemap(
            "size_t",
            c_type="size_t",
            cxx_type="size_t",
            c_header="<stddef.h>",
            cxx_header="<cstddef>",
            f_cast="int({f_var}, C_SIZE_T)",
            f_type="integer(C_SIZE_T)",
            f_kind="C_SIZE_T",
            f_module=dict(iso_c_binding=["C_SIZE_T"]),
            PY_ctor="PyInt_FromSize_t({c_deref}{c_var})",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        # XXX- sized based types for Python
        int8_t=Typemap(
            "int8_t",
            c_type="int8_t",
            cxx_type="int8_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT8_t)",
            f_type="integer(C_INT8_T)",
            f_kind="C_INT8_T",
            f_module=dict(iso_c_binding=["C_INT8_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({c_deref}{c_var})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT8",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        int16_t=Typemap(
            "int16_t",
            c_type="int16_t",
            cxx_type="int16_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT16_t)",
            f_type="integer(C_INT16_T)",
            f_kind="C_INT16_T",
            f_module=dict(iso_c_binding=["C_INT16_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({c_deref}{c_var})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT16",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        int32_t=Typemap(
            "int32_t",
            c_type="int32_t",
            cxx_type="int32_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT32_t)",
            f_type="integer(C_INT32_T)",
            f_kind="C_INT32_T",
            f_module=dict(iso_c_binding=["C_INT32_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({c_deref}{c_var})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT32",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        int64_t=Typemap(
            "int64_t",
            c_type="int64_t",
            cxx_type="int64_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT64_t)",
            f_type="integer(C_INT64_T)",
            f_kind="C_INT64_T",
            f_module=dict(iso_c_binding=["C_INT64_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({c_deref}{c_var})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT64",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        # XXX- sized based types for Python
        uint8_t=Typemap(
            "uint8_t",
            c_type="uint8_t",
            cxx_type="uint8_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT8_t)",
            f_type="integer(C_INT8_T)",
            f_kind="C_INT8_T",
            f_module=dict(iso_c_binding=["C_INT8_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({c_deref}{c_var})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT8",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        uint16_t=Typemap(
            "uint16_t",
            c_type="uint16_t",
            cxx_type="uint16_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT16_t)",
            f_type="integer(C_INT16_T)",
            f_kind="C_INT16_T",
            f_module=dict(iso_c_binding=["C_INT16_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({c_deref}{c_var})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT16",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        uint32_t=Typemap(
            "uint32_t",
            c_type="uint32_t",
            cxx_type="uint32_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT32_t)",
            f_type="integer(C_INT32_T)",
            f_kind="C_INT32_T",
            f_module=dict(iso_c_binding=["C_INT32_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({c_deref}{c_var})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT32",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        uint64_t=Typemap(
            "uint64_t",
            c_type="uint64_t",
            cxx_type="uint64_t",
            c_header="<stdint.h>",
            cxx_header="<cstdint>",
            f_cast="int({f_var}, C_INT64_t)",
            f_type="integer(C_INT64_T)",
            f_kind="C_INT64_T",
            f_module=dict(iso_c_binding=["C_INT64_T"]),
            PY_format="i",
            PY_ctor="PyInt_FromLong({c_deref}{c_var})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT64",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
        ),
        float=Typemap(
            "float",
            c_type="float",
            cxx_type="float",
            f_cast="real({f_var}, C_FLOAT)",
            f_type="real(C_FLOAT)",
            f_kind="C_FLOAT",
            f_module=dict(iso_c_binding=["C_FLOAT"]),
            PY_format="f",
            PY_ctor="PyFloat_FromDouble({c_deref}{c_var})",
            PY_get="PyFloat_AsDouble({py_var})",
            PYN_typenum="NPY_FLOAT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tonumber({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushnumber({LUA_state_var}, {c_var})",
        ),
        double=Typemap(
            "double",
            c_type="double",
            cxx_type="double",
            f_cast="real({f_var}, C_DOUBLE)",
            f_type="real(C_DOUBLE)",
            f_kind="C_DOUBLE",
            f_module=dict(iso_c_binding=["C_DOUBLE"]),
            PY_format="d",
            PY_ctor="PyFloat_FromDouble({c_deref}{c_var})",
            PY_get="PyFloat_AsDouble({py_var})",
            PYN_typenum="NPY_DOUBLE",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tonumber({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushnumber({LUA_state_var}, {c_var})",
        ),
        bool=Typemap(
            "bool",
            c_type="bool",
            cxx_type="bool",
            c_header="<stdbool.h>",
            f_type="logical",
            f_kind="C_BOOL",
            f_c_type="logical(C_BOOL)",
            f_module=dict(iso_c_binding=["C_BOOL"]),
            f_statements=dict(
                intent_in=dict(
                    c_local_var=True,
                    pre_call=["{c_var} = {f_var}  ! coerce to C_BOOL"],
                ),
                intent_out=dict(
                    c_local_var=True,
                    post_call=["{f_var} = {c_var}  ! coerce to logical"],
                ),
                intent_inout=dict(
                    c_local_var=True,
                    pre_call=["{c_var} = {f_var}  ! coerce to C_BOOL"],
                    post_call=["{f_var} = {c_var}  ! coerce to logical"],
                ),
                result=dict(
                    # The wrapper is needed to convert bool to logical
                    need_wrapper=True
                ),
            ),
            py_statements=dict(
                intent_in=dict(
                    pre_call=["bool {cxx_var} = PyObject_IsTrue({py_var});"]
                ),
                intent_inout=dict(
                    pre_call=["bool {cxx_var} = PyObject_IsTrue({py_var});"],
                    # py_var is already declared for inout
                    post_call=[
                        "{py_var} = PyBool_FromLong({c_deref}{c_var});",
                        "if ({py_var} == NULL) goto fail;",
                    ],
                    fail=[
                        "Py_XDECREF({py_var});",
                    ],
                    goto_fail=True,
                ),
                intent_out=dict(
                    decl=[
                        "{PyObject} * {py_var} = NULL;",
                    ],
                    post_call=[
                        "{py_var} = PyBool_FromLong({c_var});",
                        "if ({py_var} == NULL) goto fail;",
                    ],
                    fail=[
                        "Py_XDECREF({py_var});",
                    ],
                    goto_fail=True,
                ),
                result=dict(
                    decl=[
                        "{PyObject} * {py_var} = NULL;",
                    ],
                    post_call=[
                        "{py_var} = PyBool_FromLong({c_var});",
                        "if ({py_var} == NULL) goto fail;",
                    ],
                    fail=[
                        "Py_XDECREF({py_var});",
                    ],
                    goto_fail=True,
                ),
            ),
            # XXX PY_format='p',  # Python 3.3 or greater
            # Use py_statements.x.ctor instead of PY_ctor. This code will always be
            # added.  Older version of Python can not create a bool directly from
            # from Py_BuildValue.
            # #- PY_ctor='PyBool_FromLong({c_var})',
            PY_PyTypeObject="PyBool_Type",
            PYN_typenum="NPY_BOOL",
            LUA_type="LUA_TBOOLEAN",
            LUA_pop="lua_toboolean({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushboolean({LUA_state_var}, {c_var})",
        ),
        # implies null terminated string
        char=Typemap(
            "char",
            cxx_type="char",
            c_type="char",  # XXX - char *
            c_statements=dict(
                intent_in_buf=dict(
                    buf_args=["arg", "len_trim"],
                    cxx_local_var="pointer",
                    c_helper="ShroudStrAlloc ShroudStrFree",
                    pre_call=[
                        "char * {cxx_var} = ShroudStrAlloc(\t"
                        "{c_var},\t {c_var_trim},\t {c_var_trim});",
                    ],
                    post_call=[
                        "ShroudStrFree({cxx_var});"
                    ],
                ),
                intent_out_buf=dict(
                    buf_args=["arg", "len"],
                    c_helper="ShroudStrBlankFill",
                    post_call=[
                        "ShroudStrBlankFill({c_var}, {c_var_len});"
                    ],
                ),
                intent_inout_buf=dict(
                    buf_args=["arg", "len_trim", "len"],
                    cxx_local_var="pointer",
                    c_helper="ShroudStrAlloc ShroudStrCopy ShroudStrFree",
                    pre_call=[
                        "char * {cxx_var} = ShroudStrAlloc(\t"
                        "{c_var},\t {c_var_len},\t {c_var_trim});",
                    ],
                    post_call=[
                        # nsrc=-1 will call strlen({cxx_var})
                        "ShroudStrCopy({c_var}, {c_var_len},"
                        "\t {cxx_var},\t -1);",
                        "ShroudStrFree({cxx_var});",
                    ],
                ),
                result_buf=dict(
                    buf_args=["arg", "len"],
                    c_helper="ShroudStrCopy",
                    post_call=[
                        # nsrc=-1 will call strlen({cxx_var})
                        "ShroudStrCopy({c_var}, {c_var_len},"
                        "\t {cxx_var},\t -1);",
                    ],
                ),
                result_buf_allocatable=dict(
                    buf_args=["context"],
                    c_helper="copy_string",
                    # Copy address of result into c_var and save length.
                    # When returning a std::string (and not a reference or pointer)
                    # an intermediate object is created to save the results
                    # which will be passed to copy_string
                    post_call=[
                        "{c_var_context}->cxx.addr = {cxx_cast_to_void_ptr};",
                        "{c_var_context}->cxx.idtor = {idtor};",
                        "{c_var_context}->addr.ccharp = {cxx_var};",
                        "{c_var_context}->len = {cxx_var} == NULL ? 0 : {stdlib}strlen({cxx_var});",
                        "{c_var_context}->size = 1;",
                    ],
                ),
            ),
            f_type="character(*)",
            f_kind="C_CHAR",
            f_c_type="character(kind=C_CHAR)",
            f_c_module=dict(iso_c_binding=["C_CHAR"]),
            f_statements=dict(
                result_pure=dict(
                    need_wrapper=True,
                    f_helper="fstr_ptr",
                    call=["{F_result} = fstr_ptr({F_C_call}({F_arg_c_call}))"],
                ),
                result_allocatable=dict(
                    need_wrapper=True,
                    f_helper="copy_string",
                    post_call=[
                        "allocate(character(len={c_var_context}%len):: {f_var})",
                        "call SHROUD_copy_string_and_free"
                        "({c_var_context}, {f_var}, {c_var_context}%len)",
                    ],
                ),
            ),
            PY_format="s",
            PY_ctor="PyString_FromString({c_var})",
            LUA_type="LUA_TSTRING",
            LUA_pop="lua_tostring({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushstring({LUA_state_var}, {c_var})",
            base="string",
        ),
        # char scalar
        char_scalar=Typemap(
            "char_scalar",
            cxx_type="char",
            c_type="char",  # XXX - char *
            c_statements=dict(
                result_buf=dict(
                    buf_args=["arg", "len"],
                    c_header="<string.h>",
                    cxx_header="<cstring>",
                    post_call=[
                        "{stdlib}memset({c_var}, ' ', {c_var_len});",
                        "{c_var}[0] = {cxx_var};",
                    ],
                )
            ),
            f_type="character",
            f_kind="C_CHAR",
            f_c_type="character(kind=C_CHAR)",
            f_c_module=dict(iso_c_binding=["C_CHAR"]),
            PY_format="c",
            # #-  PY_ctor='Py_BuildValue("c", (int) {c_var})',
            PY_ctor="PyString_FromStringAndSize(&{c_var}, 1)",
            # #- PY_build_format='c',
            PY_build_arg="(int) {cxx_var}",
            LUA_type="LUA_TSTRING",
            LUA_pop="lua_tostring({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushstring({LUA_state_var}, {c_var})",
            # # base='string',
        ),
        # C++ std::string
        string=Typemap(
            "std::string",
            cxx_type="std::string",
            cxx_to_c="{cxx_var}{cxx_member}c_str()",  # cxx_member is . or ->
            c_type="char",  # XXX - char *
            impl_header="<string>",
            c_statements=dict(
                intent_in=dict(
                    cxx_local_var="scalar",
                    pre_call=["{c_const}std::string {cxx_var}({c_var});"],
                ),
                intent_out=dict(
                    cxx_header="<cstring>",
                    # #- pre_call=[
                    # #-     'int {c_var_trim} = strlen({c_var});',
                    # #-     ],
                    cxx_local_var="scalar",
                    pre_call=["{c_const}std::string {cxx_var};"],
                    post_call=[
                        # This may overwrite c_var if cxx_val is too long
                        "strcpy({c_var}, {cxx_var}{cxx_member}c_str());"
                    ],
                ),
                intent_inout=dict(
                    cxx_header="<cstring>",
                    cxx_local_var="scalar",
                    pre_call=["{c_const}std::string {cxx_var}({c_var});"],
                    post_call=[
                        # This may overwrite c_var if cxx_val is too long
                        "strcpy({c_var}, {cxx_var}{cxx_member}c_str());"
                    ],
                ),
                intent_in_buf=dict(
                    buf_args=["arg", "len_trim"],
                    cxx_local_var="scalar",
                    pre_call=[
                        (
                            "{c_const}std::string "
                            "{cxx_var}({c_var}, {c_var_trim});"
                        )
                    ],
                ),
                intent_out_buf=dict(
                    buf_args=["arg", "len"],
                    c_helper="ShroudStrCopy",
                    cxx_local_var="scalar",
                    pre_call=["std::string {cxx_var};"],
                    post_call=[
                        "ShroudStrCopy({c_var}, {c_var_len},"
                        "\t {cxx_var}{cxx_member}data(),"
                        "\t {cxx_var}{cxx_member}size());"
                    ],
                ),
                intent_inout_buf=dict(
                    buf_args=["arg", "len_trim", "len"],
                    c_helper="ShroudStrCopy",
                    cxx_local_var="scalar",
                    pre_call=["std::string {cxx_var}({c_var}, {c_var_trim});"],
                    post_call=[
                        "ShroudStrCopy({c_var}, {c_var_len},"
                        "\t {cxx_var}{cxx_member}data(),"
                        "\t {cxx_var}{cxx_member}size());"
                    ],
                ),
                result_buf=dict(
                    buf_args=["arg", "len"],
                    c_helper="ShroudStrCopy",
                    post_call=[
                        "if ({cxx_var}{cxx_member}empty()) {{+",
                        "ShroudStrCopy({c_var}, {c_var_len},"
                        "\t NULL,\t 0);",
                        "-}} else {{+",
                        "ShroudStrCopy({c_var}, {c_var_len},"
                        "\t {cxx_var}{cxx_member}data(),"
                        "\t {cxx_var}{cxx_member}size());",
                        "-}}",
                    ],
                ),

                # Uses a two part call to copy results of std::string into a
                # allocatable Fortran array.
                #    c_step1(context)
                #    allocate(character(len=context%len): Fout)
                #    c_step2(context, Fout, context%len)
                # only used with bufferifed routines and intent(out) or result
                result_buf_allocatable=dict(
                    # pass address of string and length back to Fortran
                    buf_args=["context"],
                    c_helper="copy_string",
                    # Copy address of result into c_var and save length.
                    # When returning a std::string (and not a reference or pointer)
                    # an intermediate object is created to save the results
                    # which will be passed to copy_string
                    post_call=[
                        "{c_var_context}->cxx.addr = {cxx_cast_to_void_ptr};",
                        "{c_var_context}->cxx.idtor = {idtor};",
                        "if ({cxx_var}{cxx_member}empty()) {{+",
                        "{c_var_context}->addr.ccharp = NULL;",
                        "{c_var_context}->len = 0;",
                        "-}} else {{+",
                        "{c_var_context}->addr.ccharp = {cxx_var}{cxx_member}data();",
                        "{c_var_context}->len = {cxx_var}{cxx_member}size();",
                        "-}}",
                        "{c_var_context}->size = 1;",
                    ],
                ),
            ),
            f_type="character(*)",
            f_kind="C_CHAR",
            f_c_type="character(kind=C_CHAR)",
            f_c_module=dict(iso_c_binding=["C_CHAR"]),
            f_statements=dict(
                result_pure=dict(
                    need_wrapper=True,
                    f_helper="fstr_ptr",
                    call=["{F_result} = fstr_ptr({F_C_call}({F_arg_c_call}))"],
                ),
                result_allocatable=dict(
                    need_wrapper=True,
                    f_helper="copy_string",
                    post_call=[
                        "allocate(character(len={c_var_context}%len):: {f_var})",
                        "call SHROUD_copy_string_and_free("
                        "{c_var_context}, {f_var}, {c_var_context}%len)",
                    ],
                ),
            ),
            py_statements=dict(
                intent_in=dict(
                    cxx_local_var="scalar",
                    post_parse=["{c_const}std::string {cxx_var}({c_var});"],
                ),
                intent_inout=dict(
                    cxx_local_var="scalar",
                    post_parse=["{c_const}std::string {cxx_var}({c_var});"],
                ),
                intent_out=dict(
                    cxx_local_var="scalar",
                    post_parse=["{c_const}std::string {cxx_var};"],
                ),
            ),
            PY_format="s",
            PY_ctor="PyString_FromStringAndSize(\t{cxx_var}{cxx_member}data(),\t {cxx_var}{cxx_member}size())",
            PY_build_format="s#",
            PY_build_arg="{cxx_var}{cxx_member}data(),\t {cxx_var}{cxx_member}size()",
            LUA_type="LUA_TSTRING",
            LUA_pop="lua_tostring({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushstring({LUA_state_var}, {c_var})",
            base="string",
        ),
        # C++ std::vector
        # No c_type or f_type, use attr[template]
        # C++03 "The elements of a vector are stored contiguously" (23.2.4/1).
        vector=Typemap(
            "std::vector",
            cxx_type="std::vector<{cxx_T}>",
            cxx_header="<vector>",
            # #- cxx_to_c='{cxx_var}.data()',  # C++11
            # #- cxx_to_c='{cxx_var}{cxx_member}empty() ? NULL : &{cxx_var}[0]', # C++03)
            c_statements=dict(
                intent_in_buf=dict(
                    buf_args=["arg", "size"],
                    cxx_local_var="scalar",
                    pre_call=[
                        (
                            "{c_const}std::vector<{cxx_T}> "
                            "{cxx_var}({c_var}, {c_var} + {c_var_size});"
                        )
                    ],
                ),
                # cxx_var is always a pointer to a vector
                intent_out_buf=dict(
                    buf_args=["context"],
                    cxx_local_var="pointer",
                    c_helper="capsule_data_helper copy_array",
                    pre_call=[
                        "{c_const}std::vector<{cxx_T}>"
                        "\t *{cxx_var} = new std::vector<{cxx_T}>;"
                    ],
                    post_call=[
                        # Return address and size of vector data.
                        "{c_var_context}->cxx.addr  = static_cast<void *>({cxx_var});",
                        "{c_var_context}->cxx.idtor = {idtor};",
                        "{c_var_context}->addr.cvoidp = {cxx_var}->empty()"
                        " ? NULL : &{cxx_var}->front();",
                        "{c_var_context}->len = sizeof({cxx_T});",
                        "{c_var_context}->size = {cxx_var}->size();",
                    ],
                    destructor_name="std_vector_{cxx_T}",
                    destructor=[
                        "std::vector<{cxx_T}> *cxx_ptr ="
                        " \treinterpret_cast<std::vector<{cxx_T}> *>(ptr);",
                        "delete cxx_ptr;",
                    ],
                ),
                intent_inout_buf=dict(
                    buf_args=["arg", "size", "context"],
                    cxx_local_var="pointer",
                    pre_call=[
                        "std::vector<{cxx_T}> *{cxx_var} = \tnew std::vector<{cxx_T}>\t("
                        "\t{c_var}, {c_var} + {c_var_size});"
                    ],
                    post_call=[
                        # Return address and size of vector data.
                        "{c_var_context}->cxx.addr  = static_cast<void *>({cxx_var});",
                        "{c_var_context}->cxx.idtor = {idtor};",
                        "{c_var_context}->addr.cvoidp = {cxx_var}->empty()"
                        " ? NULL : &{cxx_var}->front();",
                        "{c_var_context}->len = sizeof({cxx_T});",
                        "{c_var_context}->size = {cxx_var}->size();",
                    ],
                    destructor_name="std_vector_{cxx_T}",
                    destructor=[
                        "std::vector<{cxx_T}> *cxx_ptr ="
                        " \treinterpret_cast<std::vector<{cxx_T}> *>(ptr);",
                        "delete cxx_ptr;",
                    ],
                ),
                #                result_buf=dict(
                #                    buf_args=['arg', 'size'],
                #                    c_helper='ShroudStrCopy',
                #                    post_call=[
                #                        'if ({cxx_var}.empty()) {{+',
                #                        'ShroudStrCopy({c_var}, {c_var_len},'
                #                        'NULL, 0);',
                #                        '-}} else {{+',
                #                        'ShroudStrCopy({c_var}, {c_var_len},'
                #                        '\t {cxx_var}{cxx_member}data(),'
                #                        '\t {cxx_var}{cxx_member}size());',
                #                        '-}}',
                #                    ],
                #                ),
            ),
            f_statements=dict(
                # copy into user's existing array
                intent_out=dict(
                    f_helper="copy_array_{cxx_T}",
                    f_module=dict(iso_c_binding=["C_SIZE_T"]),
                    post_call=[
                        "call SHROUD_copy_array_{cxx_T}({c_var_context}, "
                        "{f_var}, size({f_var},kind=C_SIZE_T))"
                    ],
                ),
                intent_inout=dict(
                    f_helper="copy_array_{cxx_T}",
                    f_module=dict(iso_c_binding=["C_SIZE_T"]),
                    post_call=[
                        "call SHROUD_copy_array_{cxx_T}({c_var_context}, "
                        "{f_var}, size({f_var},kind=C_SIZE_T))"
                    ],
                ),
                # copy into allocated array
                intent_out_allocatable=dict(
                    f_helper="copy_array_{cxx_T}",
                    f_module=dict(iso_c_binding=["C_SIZE_T"]),
                    post_call=[
                        "allocate({f_var}({c_var_context}%size))",
                        "call SHROUD_copy_array_{cxx_T}({c_var_context}, "
                        "{f_var}, size({f_var},kind=C_SIZE_T))",
                    ],
                ),
                intent_inout_allocatable=dict(
                    f_helper="copy_array_{cxx_T}",
                    f_module=dict(iso_c_binding=["C_SIZE_T"]),
                    post_call=[
                        "if (allocated({f_var})) deallocate({f_var})",
                        "allocate({f_var}({c_var_context}%size))",
                        "call SHROUD_copy_array_{cxx_T}({c_var_context}, "
                        "{f_var}, size({f_var},kind=C_SIZE_T))",
                    ],
                ),
            ),
            # custom code for templates
            c_templates={
                "std::string": dict(
                    intent_in_buf=dict(
                        buf_args=["arg", "size", "len"],
                        c_helper="ShroudLenTrim",
                        cxx_local_var="scalar",
                        pre_call=[
                            "std::vector<{cxx_T}> {cxx_var};",
                            "{{+",
                            "{c_const}char * BBB = {c_var};",
                            "std::vector<{cxx_T}>::size_type",
                            "+{c_temp}i = 0,",
                            "{c_temp}n = {c_var_size};",
                            "-for(; {c_temp}i < {c_temp}n; {c_temp}i++) {{+",
                            "{cxx_var}.push_back("
                            "std::string(BBB,ShroudLenTrim(BBB, {c_var_len})));",
                            "BBB += {c_var_len};",
                            "-}}",
                            "-}}",
                        ],
                    ),
                    intent_out_buf=dict(
                        buf_args=["arg", "size", "len"],
                        c_helper="ShroudLenTrim",
                        cxx_local_var="scalar",
                        pre_call=["{c_const}std::vector<{cxx_T}> {cxx_var};"],
                        post_call=[
                            "{{+",
                            "char * BBB = {c_var};",
                            "std::vector<{cxx_T}>::size_type",
                            "+{c_temp}i = 0,",
                            "{c_temp}n = {c_var_size};",
                            "{c_temp}n = std::min({cxx_var}.size(),{c_temp}n);",
                            "-for(; {c_temp}i < {c_temp}n; {c_temp}i++) {{+",
                            "ShroudStrCopy("
                            "BBB, {c_var_len},"
                            "\t {cxx_var}[{c_temp}i].data(),"
                            "\t {cxx_var}[{c_temp}i].size());",
                            "BBB += {c_var_len};",
                            "-}}",
                            "-}}",
                        ],
                    ),
                    intent_inout_buf=dict(
                        buf_args=["arg", "size", "len"],
                        cxx_local_var="scalar",
                        pre_call=[
                            "std::vector<{cxx_T}> {cxx_var};",
                            "{{+",
                            "{c_const}char * BBB = {c_var};",
                            "std::vector<{cxx_T}>::size_type",
                            "+{c_temp}i = 0,",
                            "{c_temp}n = {c_var_size};",
                            "-for(; {c_temp}i < {c_temp}n; {c_temp}i++) {{+",
                            "{cxx_var}.push_back"
                            "(std::string(BBB,ShroudLenTrim(BBB, {c_var_len})));",
                            "BBB += {c_var_len};",
                            "-}}",
                            "-}}",
                        ],
                        post_call=[
                            "{{+",
                            "char * BBB = {c_var};",
                            "std::vector<{cxx_T}>::size_type",
                            "+{c_temp}i = 0,",
                            "{c_temp}n = {c_var_size};",
                            "-{c_temp}n = std::min({cxx_var}.size(),{c_temp}n);",
                            "for(; {c_temp}i < {c_temp}n; {c_temp}i++) {{+",
                            "ShroudStrCopy(BBB, {c_var_len},"
                            "\t {cxx_var}[{c_temp}i].data(),"
                            "\t {cxx_var}[{c_temp}i].size());",
                            "BBB += {c_var_len};",
                            "-}}",
                            "-}}",
                        ],
                    ),
                    #                    result_buf=dict(
                    #                        c_helper='ShroudStrCopy',
                    #                        post_call=[
                    #                            'if ({cxx_var}.empty()) {{+',
                    #                            'std::memset({c_var}, \' \', {c_var_len});',
                    #                            '-}} else {{+',
                    #                            'ShroudStrCopy({c_var}, {c_var_len}, '
                    #                            '\t {cxx_var}{cxx_member}data(),'
                    #                            '\t {cxx_var}{cxx_member}size());',
                    #                            '-}}',
                    #                        ],
                    #                    ),
                )
            },
            #            py_statements=dict(
            #                intent_in=dict(
            #                    cxx_local_var=True,
            #                    post_parse=[
            #                        '{c_const}std::vector<{cxx_T}> {cxx_var}({c_var});'
            #                        ],
            #                    ),
            #                ),
            #            PY_format='s',
            #            PY_ctor='PyString_FromString({c_var})',
            #            LUA_type='LUA_TSTRING',
            #            LUA_pop='lua_tostring({LUA_state_var}, {LUA_index})',
            #            LUA_push='lua_pushstring({LUA_state_var}, {c_var})',
            base="vector",
        ),
        MPI_Comm=Typemap(
            "MPI_Comm",
            cxx_type="MPI_Comm",
            c_header="mpi.h",
            cxx_header="mpi.h",
            c_type="MPI_Fint",
            # usually, MPI_Fint will be equivalent to int
            f_type="integer",
            f_kind="C_INT",
            f_c_type="integer(C_INT)",
            f_c_module=dict(iso_c_binding=["C_INT"]),
            cxx_to_c="MPI_Comm_c2f({cxx_var})",
            c_to_cxx="MPI_Comm_f2c({c_var})",
        ),
    )

    # Rename to actual types.
    # It is not possible to do dict(std::string=...)
    def_types["std::string"] = def_types["string"]
    del def_types["string"]
    def_types["std::vector"] = def_types["vector"]
    del def_types["vector"]

    set_global_types(def_types)

    return def_types


def create_enum_typemap(node):
    """Create a typemap similar to an int.
    C++ enums are converted to a C int.

    Args:
        node - EnumNode instance.
    """
    fmt_enum = node.fmtdict
    type_name = util.wformat("{namespace_scope}{enum_name}", fmt_enum)

    ntypemap = lookup_type(type_name)
    if ntypemap is None:
        inttypemap = lookup_type("int")
        ntypemap = inttypemap.clone_as(type_name)
        ntypemap.cxx_type = util.wformat(
            "{namespace_scope}{enum_name}", fmt_enum
        )
        ntypemap.c_to_cxx = util.wformat(
            "static_cast<{namespace_scope}{enum_name}>({{c_var}})", fmt_enum
        )
        ntypemap.cxx_to_c = "static_cast<int>({cxx_var})"
        ntypemap.compute_flat_name()
        register_type(type_name, ntypemap)
    return ntypemap


def create_class_typemap_from_fields(cxx_name, fields, library):
    """Create a typemap from fields.
    Used when creating typemap from YAML. (from regression/forward.yaml)
    typemap:
    - type: tutorial::Class1
      fields:
        base: shadow

    Args:
        cxx_name -
        fields - dictionary object.
        library - ast.LibraryNode.
    """
    fmt_class = library.fmtdict
    ntypemap = Typemap(cxx_name, base="shadow", cxx_type=cxx_name)
    ntypemap.update(fields)
    if ntypemap.f_module_name is None:
        raise RuntimeError(
            "typemap {} requires field f_module_name".format(cxx_name)
        )
    ntypemap.f_module = {ntypemap.f_module_name: [ntypemap.f_derived_type]}
    ntypemap.f_c_module = {
        ntypemap.f_module_name: [ntypemap.f_capsule_data_type]
    }
    fill_shadow_typemap_defaults(ntypemap, fmt_class)
    register_type(cxx_name, ntypemap)
    library.add_shadow_typemap(ntypemap)
    return ntypemap


def create_class_typemap(node, fields=None):
    """Create a typemap from a ClassNode.
    Use fields to override defaults.

    The c_type and f_capsule_data_type are a struct which contains
    a pointer to the C++ memory and information on how to delete the memory.

    Args:
        node - ast.ClassNode.
        fields - dictionary-like object.
    """
    fmt_class = node.fmtdict
    cxx_name = util.wformat("{namespace_scope}{cxx_class}", fmt_class)
    cxx_type = util.wformat("{namespace_scope}{cxx_type}", fmt_class)

    ntypemap = lookup_type(cxx_name)
    # unname = util.un_camel(name)
    f_name = fmt_class.cxx_class.lower()
    c_name = fmt_class.C_prefix + fmt_class.C_name_scope[:-1]
    ntypemap = Typemap(
        cxx_name,
        base="shadow",
        cxx_type=cxx_type,
        # XXX - look up scope for header...
        impl_header=node.cxx_header or None,
        c_type=c_name,
        f_module_name=fmt_class.F_module_name,
        f_derived_type=fmt_class.F_derived_name,
        f_capsule_data_type=fmt_class.F_capsule_data_type,
        f_module={fmt_class.F_module_name: [fmt_class.F_derived_name]},
        # #- f_to_c='{f_var}%%%s()' % fmt_class.F_name_instance_get, # XXX - develop test
        f_to_c="{f_var}%%%s" % fmt_class.F_derived_member,
    )
    # import classes which are wrapped by this module
    # XXX - deal with namespaces vs modules
    ntypemap.f_c_module = {"--import--": [ntypemap.f_capsule_data_type]}
    if fields is not None:
        ntypemap.update(fields)
    fill_shadow_typemap_defaults(ntypemap, fmt_class)
    register_type(cxx_name, ntypemap)

    fmt_class.C_type_name = ntypemap.c_type
    return ntypemap


def fill_shadow_typemap_defaults(ntypemap, fmt):
    """Add some defaults to typemap.
    When dumping typemaps to a file, only a subset is written
    since the rest are boilerplate.  This function restores
    the boilerplate.

    Args:
        ntypemap - typemap.Typemap.
        fmt -
    """
    if ntypemap.base != "shadow":
        return

    # Convert to void * to add to struct
    ntypemap.cxx_to_c = "static_cast<{c_const}void *>(\t{cxx_addr}{cxx_var})"

    # void pointer in struct -> class instance pointer
    ntypemap.c_to_cxx = (
        "static_cast<{c_const}%s *>({c_var}{c_member}addr)" % ntypemap.cxx_type
    )

    # some default for ntypemap.f_capsule_data_type
    ntypemap.f_type = "type(%s)" % ntypemap.f_derived_type
    ntypemap.f_c_type = "type(%s)" % ntypemap.f_capsule_data_type

    # XXX module name may not conflict with type name
    #    ntypemap.f_module={fmt_class.F_module_name:[unname]}

    # Return a C_capsule_data_type
    ntypemap.c_statements = dict(
        intent_in=dict(buf_args=["shadow"]),
        result=dict(
            post_call=[
                "{c_var}->addr = {cxx_cast_to_void_ptr};",
                "{c_var}->idtor = {idtor};",
            ],
        ),
    )

    # return from C function
    # f_c_return_decl='type(CPTR)' % unname,
    ntypemap.f_statements = dict(
        result=dict(
            need_wrapper=True,
            call=[
                # The c Function returns a pointer.
                # Save in a type(C_PTR) variable.
                "{F_result_ptr} = {F_C_call}({F_arg_c_call})"
            ],
        )
    )

    # The import is added in wrapf.py
    #    ntypemap.f_c_module={ '-import-': ['F_capsule_data_type']}

    ntypemap.py_statements = dict(
        intent_in=dict(
            cxx_local_var="pointer",
            post_parse=[
                "{c_const}%s * {cxx_var} ="
                "\t {py_var} ? {py_var}->{PY_type_obj} : NULL;" % ntypemap.cxx_type
            ],
        ),
        intent_inout=dict(
            cxx_local_var="pointer",
            post_parse=[
                "{c_const}%s * {cxx_var} ="
                "\t {py_var} ? {py_var}->{PY_type_obj} : NULL;" % ntypemap.cxx_type
            ],
        ),
        intent_out=dict(
            decl=[
                "{PyObject} *{py_var} = NULL;"
            ],
            post_call=[
                "{py_var} ="
                "\t PyObject_New({PyObject}, &{PyTypeObject});",
                "if ({py_var} == NULL) goto fail;",
                "{py_var}->{PY_type_obj} = {cxx_addr}{cxx_var};",
            ],
#            post_call_capsule=[
#                "{py_var}->{PY_type_dtor} = {PY_numpy_array_dtor_context} + {capsule_order};",
#            ],
            fail=[
                "Py_XDECREF({py_var});",
            ],
            goto_fail=True,
        ),
        result=dict(
#            decl=[
#                "{PyObject} *{py_var} = NULL;"
#            ],
            post_call=[
                "{PyObject} * {py_var} ="
                "\t PyObject_New({PyObject}, &{PyTypeObject});",
#                "if ({py_var} == NULL) goto fail;",
                "{py_var}->{PY_type_obj} = {cxx_addr}{cxx_var};",
            ],
#            post_call_capsule=[
#                "{py_var}->{PY_type_dtor} = {PY_numpy_array_dtor_context} + {capsule_order};",
#            ],
#            fail=[
#                "Py_XDECREF({py_var});",
#            ],
#            goto_fail=True,
        ),
    )
    # #-    if not ntypemap.PY_PyTypeObject:
    # #-        ntypemap.PY_PyTypeObject = 'UUU'
    # ntypemap.PY_ctor = 'PyObject_New({PyObject}, &{PyTypeObject})'

    ntypemap.LUA_type = "LUA_TUSERDATA"
    ntypemap.LUA_pop = (
        "\t({LUA_userdata_type} *)\t luaL_checkudata"
        '(\t{LUA_state_var}, 1, "{LUA_metadata}")'
    )
    # ntypemap.LUA_push = None  # XXX create a userdata object with metatable
    # ntypemap.LUA_statements = {}

    # allow forward declarations to avoid recursive headers
    ntypemap.forward = ntypemap.cxx_type


def create_struct_typemap(node, fields=None):
    """Create a typemap for a struct from a ClassNode.
    Use fields to override defaults.

    Args:
        node   - ast.ClassNode
        fields - dictionary-like object.
    """
    fmt_class = node.fmtdict
    cxx_name = util.wformat("{namespace_scope}{cxx_class}", fmt_class)
    cxx_type = util.wformat("{namespace_scope}{cxx_type}", fmt_class)

    # unname = util.un_camel(name)
    f_name = fmt_class.cxx_class.lower()
    c_name = fmt_class.C_prefix + f_name
    ntypemap = Typemap(
        cxx_name,
        base="struct",
        cxx_type=cxx_type,
        c_type=c_name,
        f_derived_type=fmt_class.F_derived_name,
        f_module={fmt_class.F_module_name: [fmt_class.F_derived_name]},
        PYN_descr=fmt_class.PY_struct_array_descr_variable,
    )
    if fields is not None:
        ntypemap.update(fields)
    fill_struct_typemap_defaults(node, ntypemap)
    register_type(cxx_name, ntypemap)

    fmt_class.C_type_name = ntypemap.c_type
    return ntypemap


def fill_struct_typemap_defaults(node, ntypemap):
    """Add some defaults to typemap.
    When dumping typemaps to a file, only a subset is written
    since the rest are boilerplate.  This function restores
    the boilerplate.

    Args:
        node     - ast.ClassNode
        ntypemap - typemap.Typemap.
    """
    if ntypemap.base != "struct":
        return

    helper = whelpers.add_union_helper(ntypemap.cxx_type, ntypemap.c_type)

    ntypemap.c_union = helper

    libnode = node.get_LibraryNode()
    language = libnode.language
    if language == "cxx":
        # C++ pointer -> void pointer -> C pointer
        ntypemap.cxx_to_c = (
            "static_cast<{c_const}%s *>("
            "\tstatic_cast<{c_const}void *>(\t{cxx_addr}{cxx_var}))"
            % ntypemap.c_type
        )

        # C pointer -> void pointer -> C++ pointer
        ntypemap.c_to_cxx = (
            "static_cast<{c_const}%s *>("
            "\tstatic_cast<{c_const}void *>(\t{c_var}))" % ntypemap.cxx_type
        )
    else:  # language == "c"
        # The struct from the user's library is used.
        ntypemap.c_header = libnode.cxx_header
        ntypemap.c_type = ntypemap.cxx_type

    # To convert, extract correct field from union
    # #-    ntypemap.cxx_to_c = '{cxx_addr}{cxx_var}.cxx'
    # #-    ntypemap.c_to_cxx = '{cxx_addr}{cxx_var}.c'

    ntypemap.f_type = "type(%s)" % ntypemap.f_derived_type

    # XXX module name may not conflict with type name
    # #-    ntypemap.f_module = {fmt_class.F_module_name:[unname]}

    ntypemap.c_statements = dict(result=dict(c_helper=helper))

    # #-    ntypemap.PYN_typenum = 'NPY_VOID'
    # #-    if not ntypemap.PY_PyTypeObject:
    # #-        ntypemap.PY_PyTypeObject = 'UUU'
    # ntypemap.PY_ctor = 'PyObject_New({PyObject}, &{PyTypeObject})'

    ntypemap.LUA_type = "LUA_TUSERDATA"
    ntypemap.LUA_pop = (
        "\t({LUA_userdata_type} *)\t luaL_checkudata"
        '(\t{LUA_state_var}, 1, "{LUA_metadata}")'
    )
    # ntypemap.LUA_push = None  # XXX create a userdata object with metatable
    # ntypemap.LUA_statements = {}


def lookup_c_statements(arg):
    """Look up the c_statements for an argument.
    If the argument type is a template, look for
    template specific c_statements.

    Args:
        arg -
    """
    arg_typemap = arg.typemap

    c_statements = arg_typemap.c_statements
    if arg.template_arguments:
        base_typemap = arg_typemap
        arg_typemap = arg.template_arguments[0].typemap
        cxx_T = arg_typemap.name
        c_statements = base_typemap.c_templates.get(cxx_T, c_statements)
    return arg_typemap, c_statements

empty_stmts = {}
def lookup_stmts(stmts, path):
    """
    Lookup path in stmts.
    Used to find specific cases first, then fall back to general.
    ex path = ['result', 'allocatable']
         Finds 'result_allocatable' if it exists, else 'result'.
    If not found, return an empty dictionary.

    path typically consists of:
      intent_in, intent_out, intent_inout, result
      generated_clause - buf
      deref - allocatable

    Args:
        stmts - dictionary
        path  - list of name components.
                Blank entries are ignored.
    """
    work = [ part for part in path if part ] # skip empty components
    while work:
        check = '_'.join(work)
        if check in stmts:
            return stmts[check]
        work.pop()
    return empty_stmts
        
def compute_name(path, char="_"):
    """
    Compute a name from a list of components.
    Blank entries are filtered out.

    Args:
        path  - list of name components.
    """
    work = [ part for part in path if part ] # skip empty components
    return char.join(work)


def update_for_language(stmts, lang):
    """
    Move language specific entries to current language.

    stmts=dict(
      foo_bar=dict(
        c_decl=[],
        cxx_decl=[],
      )
    )

    For lang==c,
      foo_bar["decl"] = foo_bar["c_decl"]
    """
    for item in stmts.values():
        for clause in ["decl", "post_parse", "pre_call", "post_call",
                       "cleanup", "fail"]:
            specific = lang + "_" + clause
            if specific in item:
                item[clause] = item[specific]
