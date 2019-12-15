# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
Create and manage typemaps used to convert between languages.

buf_args documented in cwrapper.rst.
"""
from __future__ import print_function

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

# The tree of c and fortran statements
cf_tree = {}

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
        ("c_return_code", None),
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
        # Lua
        ("LUA_type", "LUA_TNONE"),
        ("LUA_pop", "POP"),
        ("LUA_push", "PUSH"),
        ("LUA_statements", {}),
        ("sgroup", "unknown"),  # statement group. ex. native, string, vector
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
            sgroup="native",
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
            sgroup="native",
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
            sgroup="native",
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
            sgroup="native",
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
            sgroup="native",
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
            sgroup="native",
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
            sgroup="native",
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
            sgroup="native",
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
            PY_format="n",
            PY_ctor="PyInt_FromSize_t({c_deref}{c_var})",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
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
            sgroup="native",
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
            sgroup="native",
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
            sgroup="native",
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
            PY_format="L",
            PY_ctor="PyInt_FromLong({c_deref}{c_var})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT64",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
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
            sgroup="native",
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
            sgroup="native",
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
            sgroup="native",
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
            PY_format="L",
            PY_ctor="PyInt_FromLong({c_deref}{c_var})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT64",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
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
            sgroup="native",
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
            sgroup="native",
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
            base="bool",
            sgroup="bool",
        ),
        # implies null terminated string
        char=Typemap(
            "char",
            cxx_type="char",
            c_type="char",  # XXX - char *
            f_type="character(*)",
            f_kind="C_CHAR",
            f_c_type="character(kind=C_CHAR)",
            f_c_module=dict(iso_c_binding=["C_CHAR"]),
            PY_format="s",
            PY_ctor="PyString_FromString({c_var})",
            LUA_type="LUA_TSTRING",
            LUA_pop="lua_tostring({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushstring({LUA_state_var}, {c_var})",
            base="string",
            sgroup="char",
        ),
        # char scalar
        char_scalar=Typemap(
            "char_scalar",
            cxx_type="char",
            c_type="char",  # XXX - char *
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
            sgroup='schar',
        ),
        # C++ std::string
        string=Typemap(
            "std::string",
            cxx_type="std::string",
            cxx_to_c="{cxx_var}{cxx_member}c_str()",  # cxx_member is . or ->
            c_type="char",  # XXX - char *
            impl_header="<string>",
            f_type="character(*)",
            f_kind="C_CHAR",
            f_c_type="character(kind=C_CHAR)",
            f_c_module=dict(iso_c_binding=["C_CHAR"]),
            PY_format="s",
            PY_ctor="PyString_FromStringAndSize(\t{cxx_var}{cxx_member}data(),\t {cxx_var}{cxx_member}size())",
            PY_build_format="s#",
            PY_build_arg="{cxx_var}{cxx_member}data(),\t {cxx_var}{cxx_member}size()",
            LUA_type="LUA_TSTRING",
            LUA_pop="lua_tostring({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushstring({LUA_state_var}, {c_var})",
            base="string",
            sgroup="string",
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
            # custom code for templates
            #            PY_format='s',
            #            PY_ctor='PyString_FromString({c_var})',
            #            LUA_type='LUA_TSTRING',
            #            LUA_pop='lua_tostring({LUA_state_var}, {LUA_index})',
            #            LUA_push='lua_pushstring({LUA_state_var}, {c_var})',
            base="vector",
            sgroup="vector",
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

def update_typemap_for_language(language):
    """Preprocess statements for lookup.

    Update statements for c or c++.
    Fill in cf_tree.
    """
    update_for_language(fc_statements, language)
    update_stmt_tree(fc_statements, cf_tree)

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
    ntypemap = Typemap(cxx_name, base="shadow", sgroup="shadow", cxx_type=cxx_name)
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
        sgroup="shadow",
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

    # return from C function
    # f_c_return_decl='type(CPTR)' % unname,

    # The import is added in wrapf.py
    #    ntypemap.f_c_module={ '-import-': ['F_capsule_data_type']}

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
        sgroup="struct",
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

    libnode = node.get_LibraryNode()
    language = libnode.language
    if language == "c":
        # The struct from the user's library is used.
        ntypemap.c_header = libnode.cxx_header
        ntypemap.c_type = ntypemap.cxx_type

    # To convert, extract correct field from union
    # #-    ntypemap.cxx_to_c = '{cxx_addr}{cxx_var}.cxx'
    # #-    ntypemap.c_to_cxx = '{cxx_addr}{cxx_var}.c'

    ntypemap.f_type = "type(%s)" % ntypemap.f_derived_type

    # XXX module name may not conflict with type name
    # #-    ntypemap.f_module = {fmt_class.F_module_name:[unname]}

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
    template specialization.

    Args:
        arg -
    """
    arg_typemap = arg.typemap

    specialize = []
    if arg.template_arguments:
        arg_typemap = arg.template_arguments[0].typemap
        specialize.append(arg_typemap.sgroup)
    return arg_typemap, specialize

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
        
def lookup_fc_stmts(path):
    return lookup_stmts_tree(cf_tree, path)
        
def compute_name(path, char="_"):
    """
    Compute a name from a list of components.
    Blank entries are filtered out.

    Used to find C_error_pattern.
    
    Args:
        path  - list of name components.
    """
    work = [ part for part in path if part ] # skip empty components
    return char.join(work)


def create_buf_variable_names(options, blk, attrs, c_var):
    """Define variable names for buffer arguments.
    If user has not explicitly set, then compute from option template.
    """
    for buf_arg in blk.get("buf_args", []):
        if buf_arg in attrs:
            # do not override user specified variable name
            continue
        if buf_arg == "size":
            attrs["size"] = options.C_var_size_template.format(
                c_var=c_var
            )
        elif buf_arg == "capsule":
            attrs["capsule"] = options.C_var_capsule_template.format(
                c_var=c_var
            )
        elif buf_arg == "context":
            attrs["context"] = options.C_var_context_template.format(
                c_var=c_var
            )
        elif buf_arg == "len_trim":
            attrs["len_trim"] = options.C_var_trim_template.format(
                c_var=c_var
            )
        elif buf_arg == "len":
            attrs["len"] = options.C_var_len_template.format(
                c_var=c_var
            )


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
                # XXX - maybe make sure clause does not already exist.
                item[clause] = item[specific]


def update_stmt_tree(stmts, tree):
    """Update tree by adding stmts.  Each key in stmts is split by
    underscore then inserted into tree to form nested dictionaries to
    the values from stmts.  The end key is named _node, since it is
    impossible to have an intermediate element with that name (since
    they're split on underscore).

    stmts = {"c_native_in":1,
             "c_native_out":2,
             "c_native_pointer_out":3,
             "c_string_in":4}
    tree = {
      "c": {
         "native": {
           "in": {"_node":1},
           "out":{"_node":2},
           "pointer":{
             "out":{"_node":3},
           },
         },
         "string":{
           "in": {"_node":4},
         },
      },
    }
    """
    for key, node in stmts.items():
        step = tree
        steps = key.split("_")
        for part in steps:
            step = step.setdefault(part, {})
        step['_node'] = node
        node["key"] = key  # useful for debugging


def lookup_stmts_tree(tree, path):
    """
    Lookup path in statements tree.
    Look for longest path which matches.
    Used to find specific cases first, then fall back to general.
    ex path = ['result', 'allocatable']
         Finds 'result_allocatable' if it exists, else 'result'.
    If not found, return an empty dictionary.

    path typically consists of:
      in, out, inout, result
      generated_clause - buf
      deref - allocatable

    Args:
        tree  - dictionary of nested dictionaries
        path  - list of name components.
                Blank entries are ignored.
    """
    found = empty_stmts
    work = []
    step = tree
    for part in path:
        if not part:
            # skip empty parts
            continue
        if part in step:
            step = step[part]
            found = step.get("_node", found)
    return found


                
# language   "c"    
# sgroup     "native", "string", "char"
# spointer   "pointer" ""
# intent     "in", "out", "inout", "result"
# generated  "buf"
# attribute  "allocatable"
#
# language   "f"
# sgroup     "native", "string", "char"
# spointer   "pointer" ""
# intent     "in", "out", "inout", "result"
# attribute  "allocatable"

fc_statements = dict(
    f_bool_in=dict(
        c_local_var=True,
        pre_call=["{c_var} = {f_var}  ! coerce to C_BOOL"],
    ),
    f_bool_out=dict(
        c_local_var=True,
        post_call=["{f_var} = {c_var}  ! coerce to logical"],
    ),
    f_bool_inout=dict(
        c_local_var=True,
        pre_call=["{c_var} = {f_var}  ! coerce to C_BOOL"],
        post_call=["{f_var} = {c_var}  ! coerce to logical"],
    ),
    f_bool_result=dict(
        # The wrapper is needed to convert bool to logical
        need_wrapper=True
    ),

    # Function has a result with deref(allocatable).
    #
    #    C wrapper:
    #       Add context argument for result
    #       Fill in values to describe array.
    #
    #    Fortran:
    #        c_step1(context)
    #        allocate(Fout(len))
    #        c_step2(context, Fout, size(len))
    c_native_result_buf_pointer=dict(
        buf_args=["context"],
        c_helper="array_context copy_array",
        post_call=[
            "{c_var_context}->cxx.addr  = {cxx_var};",
            "{c_var_context}->cxx.idtor = {idtor};",
            "{c_var_context}->addr.cvoidp = {cxx_var};",
            "{c_var_context}->len = sizeof({cxx_type});",
            "{c_var_context}->size = *{c_var_dimension};",
        ],
    ),
    f_native_result_allocatable_pointer=dict(
        buf_args=["context"],
        f_helper="array_context copy_array_{cxx_type}",
        post_call=[
            # XXX - allocate scalar
            "allocate({f_var}({c_var_dimension}))",
            "call SHROUD_copy_array_{cxx_type}"
            "({c_var_context}, {f_var}, size({f_var}, kind=C_SIZE_T))",
        ],
    ),

    c_char_in_buf=dict(
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
    c_char_out_buf=dict(
        buf_args=["arg", "len"],
        c_helper="ShroudStrBlankFill",
        post_call=[
            "ShroudStrBlankFill({c_var}, {c_var_len});"
        ],
    ),
    c_char_inout_buf=dict(
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
    c_char_result_buf=dict(
        buf_args=["arg", "len"],
        c_helper="ShroudStrCopy",
        post_call=[
            # nsrc=-1 will call strlen({cxx_var})
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {cxx_var},\t -1);",
        ],
    ),
    c_char_result_buf_allocatable=dict(
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
    f_char_result_allocatable=dict(
        need_wrapper=True,
        f_helper="copy_string",
        post_call=[
            "allocate(character(len={c_var_context}%len):: {f_var})",
            "call SHROUD_copy_string_and_free"
            "({c_var_context}, {f_var}, {c_var_context}%len)",
        ],
    ),

    c_schar_result_buf=dict(
        buf_args=["arg", "len"],
        c_header="<string.h>",
        cxx_header="<cstring>",
        post_call=[
            "{stdlib}memset({c_var}, ' ', {c_var_len});",
            "{c_var}[0] = {cxx_var};",
        ],
    ),

    c_string_in=dict(
        cxx_local_var="scalar",
        pre_call=["{c_const}std::string {cxx_var}({c_var});"],
    ),
    c_string_out=dict(
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
    c_string_inout=dict(
        cxx_header="<cstring>",
        cxx_local_var="scalar",
        pre_call=["{c_const}std::string {cxx_var}({c_var});"],
        post_call=[
            # This may overwrite c_var if cxx_val is too long
            "strcpy({c_var}, {cxx_var}{cxx_member}c_str());"
        ],
    ),
    c_string_in_buf=dict(
        buf_args=["arg", "len_trim"],
        cxx_local_var="scalar",
        pre_call=[
            (
                "{c_const}std::string "
                "{cxx_var}({c_var}, {c_var_trim});"
            )
        ],
    ),
    c_string_out_buf=dict(
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
    c_string_inout_buf=dict(
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
    c_string_result_buf=dict(
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
    # std::string * function()
    c_string_result_buf_allocatable=dict(
        # pass address of string and length back to Fortran
        buf_args=["context"],
        c_helper="copy_string ShroudStrToArray",
        # Copy address of result into c_var and save length.
        # When returning a std::string (and not a reference or pointer)
        # an intermediate object is created to save the results
        # which will be passed to copy_string
        post_call=[
            "ShroudStrToArray({c_var_context}, {cxx_addr}{cxx_var}, {idtor});",
        ],
    ),
    # std::string function()
    # Must allocate the std::string then assign to it via cxx_rv_decl.
    # This allows the std::string to outlast the function return.
    c_string_result_buf_allocatable_scalar=dict(
        # pass address of string and length back to Fortran
        buf_args=["context"],
        #                    cxx_local_var="pointer",
        c_helper="copy_string ShroudStrToArray",
        # Copy address of result into c_var and save length.
        # When returning a std::string (and not a reference or pointer)
        # an intermediate object is created to save the results
        # which will be passed to copy_string
        pre_call=[
            "std::string * {cxx_var} = new std::string;",
        ],
        destructor_name="new_string",
        destructor=[
            "std::string *cxx_ptr = \treinterpret_cast<std::string *>(ptr);",
            "delete cxx_ptr;",
        ],
        post_call=[
            "ShroudStrToArray({c_var_context}, {cxx_var}, {idtor});",
        ],
    ),
    
    f_string_result_allocatable=dict(
        need_wrapper=True,
        f_helper="copy_string",
        post_call=[
            "allocate(character(len={c_var_context}%len):: {f_var})",
            "call SHROUD_copy_string_and_free("
            "{c_var_context}, {f_var}, {c_var_context}%len)",
        ],
    ),
    
    
    c_vector_in_buf=dict(
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
    c_vector_out_buf=dict(
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
    c_vector_inout_buf=dict(
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
    # Same as intent_out_buf.
    c_vector_result_buf=dict(
        buf_args=["context"],
        #                    cxx_local_var="pointer",
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
    #                c_vector_result_buf=dict(
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
    
    # Specialize for vector<string>.
    c_vector_in_buf_string=dict(
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
    c_vector_out_buf_string=dict(
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
    c_vector_inout_buf_string=dict(
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
    #                    c_vector_result_buf_string=dict(
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
    # copy into user's existing array
    f_vector_out=dict(
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "call SHROUD_copy_array_{cxx_T}({c_var_context}, "
            "{f_var}, size({f_var},kind=C_SIZE_T))"
        ],
    ),
    f_vector_inout=dict(
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "call SHROUD_copy_array_{cxx_T}({c_var_context}, "
            "{f_var}, size({f_var},kind=C_SIZE_T))"
        ],
    ),
    f_vector_result=dict(
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "call SHROUD_copy_array_{cxx_T}({c_var_context}, "
            "{f_var}, size({f_var},kind=C_SIZE_T))"
        ],
    ),
    # copy into allocated array
    f_vector_out_allocatable=dict(
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "allocate({f_var}({c_var_context}%size))",
            "call SHROUD_copy_array_{cxx_T}({c_var_context}, "
            "{f_var}, size({f_var},kind=C_SIZE_T))",
        ],
    ),
    f_vector_inout_allocatable=dict(
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "if (allocated({f_var})) deallocate({f_var})",
            "allocate({f_var}({c_var_context}%size))",
            "call SHROUD_copy_array_{cxx_T}({c_var_context}, "
            "{f_var}, size({f_var},kind=C_SIZE_T))",
        ],
    ),
    f_vector_result_allocatable=dict(   # same as intent_out
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "allocate({f_var}({c_var_context}%size))",
            "call SHROUD_copy_array_{cxx_T}({c_var_context}, "
            "{f_var}, size({f_var},kind=C_SIZE_T))",
        ],
    ),

    # Return a C_capsule_data_type
    c_shadow_in=dict(
        buf_args=["shadow"]
    ),
    c_shadow_result=dict(
        post_call=[
            "{c_var}->addr = {cxx_cast_to_void_ptr};",
            "{c_var}->idtor = {idtor};",
        ],
    ),
    f_shadow_result=dict(
        need_wrapper=True,
        call=[
            # The c Function returns a pointer.
            # Save in a type(C_PTR) variable.
            "{F_result_ptr} = {F_C_call}({F_arg_c_call})"
        ],
    ),

    c_struct_in=dict(
        # C pointer -> void pointer -> C++ pointer
        cxx_local_var="pointer",
        cxx_pre_call=[
            "{c_const}{cxx_type} * {cxx_var} = \tstatic_cast<{c_const}{cxx_type} *>\t(static_cast<{c_const}void *>(\t{c_addr}{c_var}));",
        ],
    ),
    c_struct_out=dict(
        cxx_local_var="pointer",
        cxx_pre_call=[
            "{c_const}{cxx_type} * {cxx_var} = \tstatic_cast<{c_const}{cxx_type} *>\t(static_cast<{c_const}void *>(\t{c_addr}{c_var}));",
        ],
    ),
    c_struct_inout=dict(
        cxx_local_var="pointer",
        cxx_pre_call=[
            "{c_const}{cxx_type} * {cxx_var} = \tstatic_cast<{c_const}{cxx_type} *>\t(static_cast<{c_const}void *>(\t{c_addr}{c_var}));",
        ],
    ),
    c_struct_result=dict(
        # C++ pointer -> void pointer -> C pointer
        c_local_var="pointer",
        cxx_post_call=[
            "{c_const}{c_type} * {c_var} = \tstatic_cast<{c_const}{c_type} *>(\tstatic_cast<{c_const}void *>(\t{cxx_addr}{cxx_var}));",
        ],
    ),
)
                
