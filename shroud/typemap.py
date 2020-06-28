# Copyright (c) 2017-2020, Lawrence Livermore National Security, LLC and
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

# The tree of c and fortran statements.
cf_tree = {}
default_scopes = dict()

class Typemap(object):
    """Collect fields for an argument.
    This used to be a dict but a class has better access semantics:
       i.attr vs d['attr']
    It also initializes default values to avoid  d.get('attr', default)

    c_header and cxx_header are used for interface. For example,
    size_t uses <stddef.h> and <cstddef>.

    impl_header is used for implementation, i.e. the wrap.cpp file.
    For example, std::string uses <string>. <string> should not be in
    the interface since the wrapper is a C API.

    wrap_header is used for generated wrappers for shadow classes.
    """

    # Array of known keys with default values
    _order = (
        ("flat_name", None),  # Name when used by wrapper identifiers
        ("template_suffix", None),  # Name when used by wrapper identifiers
                                    # when added to class/struct format.
                    # Set from format.template_suffix in YAML for class.
        ("base", "unknown"),  # Base type: 'string'
        ("forward", None),  # Forward declaration
        ("typedef", None),  # Initialize from existing type
        ("cpp_if", None),  # C preprocessor test for c_header
        ("idtor", "0"),  # index of capsule_data destructor
        ("cxx_type", None),  # Name of type in C++, including namespace
        ("cxx_to_c", None),  # Expression to convert from C++ to C
        # None implies {cxx_var} i.e. no conversion
        (
            "cxx_header",
            [],
        ),  # Name of C++ header file required for implementation
        # For example, if cxx_to_c was a function
        ("c_type", None),  # Name of type in C
        ("c_header", []),  # Name of C header file required for type
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
        ("f_module", None),  # Fortran modules needed for type  (dictionary)
        ("f_cast", "{f_var}"),  # Expression to convert to type
                                # e.g. intrinsics such as INT and REAL.
        ("impl_header", []), # implementation header
        ("wrap_header", []), # generated wrapper header
        # Python
        ("PY_format", "O"),  # 'format unit' for PyArg_Parse
        ("PY_PyTypeObject", None),  # variable name of PyTypeObject instance
        ("PY_PyObject", None),  # typedef name of PyObject instance
        ("PY_ctor", None),  # expression to create object.
        # ex. PyFloat_FromDouble({c_deref}{c_var})
        ("PY_get", None),  # expression to create type from PyObject.
        # ex. PyFloat_AsDouble({py_var})
        # Name of converter function with prototype (PyObject *, void *).
        ("PY_to_object", None),  # PyBuild - object=converter(address)
        (
            "PY_from_object",
            None,
        ),  # PyArg_Parse - status=converter(object, address);
        ("PY_to_object_idtor", None),  # object=converter(address, idtor)
        ("PY_build_arg", None),  # argument for Py_BuildValue
        ("PY_build_format", None),  # 'format unit' for Py_BuildValue
        ("PY_struct_as", None),  # For struct - "class" or "list"
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
        ("sh_type", "SH_TYPE_OTHER"),
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
        for key, value in dct.items():
            if key in ["c_header", "cxx_header", "impl_header", "wrap_header"]:
                # Blank delimited strings to list
                if isinstance(value,list):
                    setattr(self, key, value)
                else:
                    setattr(self, key, value.split())
            elif key in self.defaults:
                setattr(self, key, value)
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

    def __export_yaml__(self, output, mode="all"):
        """Write out a subset of a wrapped type.
        Other fields are set with fill_shadow_typemap_defaults.

        Args:
            output -
        """
        # Temporary dictionary to allow convert on header fields.
        if mode == "all":
            order = self._keyorder
        else: # class
            # To be used by other libraries which import shadow types.
            if self.base == "shadow":
                order = [
                    "base",
                    "wrap_header",
                ]
            else:
                order = [
                    "base",
                    "cxx_header",
                    "c_header",
                ]
            order.extend([
#                "cxx_type",  # same as the dict key
                "c_type",
                "f_module_name",
                "f_derived_type",
                "f_capsule_data_type",
                "f_to_c",
            ])
                
        util.as_yaml(self, order, output)


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
            f_c_module=dict(iso_c_binding=["C_PTR"]),
            PY_ctor="PyCapsule_New({ctor_expr}, NULL, NULL)",
            sh_type="SH_TYPE_CPTR",
            base="void",
            sgroup="void",
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
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_SHORT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_SHORT",
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
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_INT",
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
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_LONG",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_LONG",
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
            # #- PY_ctor='PyInt_FromLong({ctor_expr})',
            PYN_typenum="NPY_LONGLONG",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_LONG_LONG",
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
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_SHORT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_UNSIGNED_SHORT",
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
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_UNSIGNED_INT",
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
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_LONG",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_UNSIGNED_LONG",
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
            # #- PY_ctor='PyInt_FromLong({ctor_expr})',
            PYN_typenum="NPY_LONGLONG",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_UNSIGNED_LONG_LONG",
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
            PY_ctor="PyInt_FromSize_t({ctor_expr})",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_SIZE_T",
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
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT8",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_INT8_T",
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
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT16",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_INT16_T",
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
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT32",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_INT32_T",
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
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_INT64",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_INT64_T",
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
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT8",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_UINT8_T",
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
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT16",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_UINT16_T",
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
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT32",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_UINT32_T",
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
            PY_ctor="PyInt_FromLong({ctor_expr})",
            PY_get="PyInt_AsLong({py_var})",
            PYN_typenum="NPY_UINT64",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tointeger({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushinteger({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_UINT64_T",
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
            PY_ctor="PyFloat_FromDouble({ctor_expr})",
            PY_get="PyFloat_AsDouble({py_var})",
            PYN_typenum="NPY_FLOAT",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tonumber({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushnumber({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_FLOAT",
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
            PY_ctor="PyFloat_FromDouble({ctor_expr})",
            PY_get="PyFloat_AsDouble({py_var})",
            PYN_typenum="NPY_DOUBLE",
            LUA_type="LUA_TNUMBER",
            LUA_pop="lua_tonumber({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushnumber({LUA_state_var}, {c_var})",
            sgroup="native",
            sh_type="SH_TYPE_DOUBLE",
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
            sh_type="SH_TYPE_BOOL",
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
            PY_ctor="PyString_FromString({ctor_expr})",
#            PY_get="PyString_AsString({py_var})",
            PYN_typenum="NPY_INTP",  # void *    # XXX - 
            LUA_type="LUA_TSTRING",
            LUA_pop="lua_tostring({LUA_state_var}, {LUA_index})",
            LUA_push="lua_pushstring({LUA_state_var}, {c_var})",
            base="string",
            sgroup="char",
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
            PY_ctor="PyString_FromStringAndSize({ctor_expr})",
            PY_build_format="s#",
            # XXX need cast after PY_SSIZE_T_CLEAN
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
    update_stmt_tree(fc_statements, cf_tree, default_stmts)

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
        impl_header=node.find_header(),
        wrap_header=fmt_class.C_header_utility,
        c_type=c_name,
        f_module_name=fmt_class.F_module_name,
        f_derived_type=fmt_class.F_derived_name,
        f_capsule_data_type=fmt_class.F_capsule_data_type,
        f_module={fmt_class.F_module_name: [fmt_class.F_derived_name]},
        # #- f_to_c='{f_var}%%%s()' % fmt_class.F_name_instance_get, # XXX - develop test
        f_to_c="{f_var}%%%s" % fmt_class.F_derived_member,
        sh_type="SH_TYPE_OTHER",
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

    # Convert to void * to add to context struct
    ntypemap.cxx_to_c = "static_cast<{c_const}void *>(\t{cxx_addr}{cxx_var})"

    # void pointer in struct -> class instance pointer
    ntypemap.c_to_cxx = (
        "static_cast<{c_const}%s *>\t({c_var}->addr)" % ntypemap.cxx_type
    )

    # some default for ntypemap.f_capsule_data_type
    ntypemap.f_type = "type(%s)" % ntypemap.f_derived_type
    ntypemap.f_c_type = "type(%s)" % ntypemap.f_capsule_data_type

    # XXX module name may not conflict with type name
    #    ntypemap.f_module={fmt_class.F_module_name:[unname]}

    # return from C function
    # f_c_return_decl='type(C_PTR)' % unname,

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
        f_c_module={"--import--": [fmt_class.F_derived_name]},
        PYN_descr=fmt_class.PY_struct_array_descr_variable,
        sh_type="SH_TYPE_STRUCT",
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
        # XXX - if struct in class, uses class.cxx_header?
        ntypemap.c_header = libnode.cxx_header
        ntypemap.c_type = ntypemap.cxx_type

    # To convert, extract correct field from union
    # #-    ntypemap.cxx_to_c = '{cxx_addr}{cxx_var}.cxx'
    # #-    ntypemap.c_to_cxx = '{cxx_addr}{cxx_var}.c'

    ntypemap.PY_struct_as = node.options.PY_struct_arg
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

def lookup_local_stmts(path, parent, node):
    """Look in node.fstatements for additional statements.
    XXX - Only used with result.
    mode - "update", "replace"

    Args:
        path   - list of path components ["c", "buf"]
        parent - parent Scope.
        node   - FunctionNode.
    """
    name = compute_name(path)
    blk = node.fstatements.get(name, None)
    if blk:
        mode = blk.get("mode", "update")
        if mode == "update":
            blk.reparent(parent)
            return blk
    return parent

def create_buf_variable_names(options, blk, attrs):
    """Turn on attribute for buf_arg if defined in blk.
    """
    for buf_arg in blk.buf_args:
        if attrs[buf_arg] is not None and \
           attrs[buf_arg] is not True:
            # None - Not set.
            # True - Do not override user specified variable name.
            pass
        elif buf_arg in ["size", "capsule", "context",
                         "len_trim", "len"]:
            attrs[buf_arg] = True

def set_buf_variable_names(options, attrs, c_var):
    """Set attribute name from option template.
    XXX - make sure they don't conflict with other names.
    """
    if attrs["size"] is True:
        attrs["size"] = options.C_var_size_template.format(
            c_var=c_var
        )
    if attrs["capsule"] is True:
        attrs["capsule"] = options.C_var_capsule_template.format(
            c_var=c_var
        )
    if attrs["owner"] == "caller" and \
       attrs["deref"] == "pointer" \
              and attrs["capsule"] is None:
        attrs["capsule"] = options.C_var_capsule_template.format(
            c_var=c_var
        )
    if attrs["context"] is True:
        attrs["context"] = options.C_var_context_template.format(
            c_var=c_var
        )
    if attrs["cdesc"] is True:
        # XXX - not sure about future of cdesc and difference with context.
        attrs["context"] = options.C_var_context_template.format(
            c_var=c_var
        )
    if attrs["len_trim"] is True:
        attrs["len_trim"] = options.C_var_trim_template.format(
            c_var=c_var
        )
    if attrs["len"] is True:
        attrs["len"] = options.C_var_len_template.format(
            c_var=c_var
        )

def assign_buf_variable_names(attrs, fmt):
    """
    Transfer names from attribute to fmt.
    """
    if attrs["capsule"]:
        fmt.c_var_capsule = attrs["capsule"]
    if attrs["context"]:
        fmt.c_var_context = attrs["context"]
    if attrs["len"]:
        fmt.c_var_len = attrs["len"]
    if attrs["len_trim"]:
        fmt.c_var_trim = attrs["len_trim"]
    if attrs["size"]:
        fmt.c_var_size = attrs["size"]
            

def compute_return_prefix(arg, local_var):
    """Compute how to access variable: dereference, address, as-is"""
    if local_var == "scalar":
        if arg.is_pointer():
            return "&"
        else:
            return ""
    elif local_var == "pointer":
        if arg.is_pointer():
            return ""
        else:
            return "*"
    elif local_var == "funcptr":
        return ""
    elif arg.is_reference():
        # Convert a return reference into a pointer.
        return "&"
    else:
        return ""

def update_for_language(stmts, lang):
    """
    Move language specific entries to current language.

    stmts=[
      dict(
        name='foo_bar',
        c_declare=[],
        cxx_declare=[],
      ),
      ...
    ]

    For lang==c,
      foo_bar["declare"] = foo_bar["c_declare"]
    """
    for item in stmts:
        for clause in ["cxx_local_var", "declare", "post_parse",
                       "pre_call", "post_call",
                       "cleanup", "fail"]:
            specific = lang + "_" + clause
            if specific in item:
                # XXX - maybe make sure clause does not already exist.
                item[clause] = item[specific]


def update_stmt_tree(stmts, tree, defaults):
    """Update tree by adding stmts.  Each key in stmts is split by
    underscore then inserted into tree to form nested dictionaries to
    the values from stmts.  The end key is named _node, since it is
    impossible to have an intermediate element with that name (since
    they're split on underscore).

    Implement "base" field.  Base must be defined before use.

    Add "_key" to tree to aid debugging.

    Each typemap is converted into a Scope instance with the parent
    based based on the language (c or f) and added as "scope" field.
    This additional layer of indirection is needed to implement base.

    stmts = [
       {name="c_native_in",}           # value1
       {name="c_native_out",}          # value2
       {name="c_native_pointer_out",}  # value3
       {name="c_string_in",}           # value4
    ]
    tree = {
      "c": {
         "native": {
           "in": {"_node":value1},
           "out":{"_node":value2},
           "pointer":{
             "out":{"_node":value3},
           },
         },
         "string":{
           "in": {"_node":value4},
         },
      },
    }

    """
    # Convert defaults into Scope nodes.
    for key, node in defaults.items():
        default_scopes[key] = node()

    # index by name to find aliases
    # XXX - look for duplicate names?
    nodes = {}
    for node in stmts:
        if "name" not in node:
            raise RuntimeError("Missing name in statements: {}".
                               format(str(node)))
        if node["name"] in nodes:
            raise RuntimeError("Duplicate key in statements: {}".
                               format(node["name"]))
        nodes[node["name"]] = node

    for node in stmts:
        key = node["name"]
        step = tree
        steps = key.split("_")
        label = []
        for part in steps:
            step = step.setdefault(part, {})
            label.append(part)
            step["_key"] = "_".join(label)
#        if "alias" in node:
#            step['_node'] = nodes[node["alias"]]
        if "base" in node:
            step['_node'] = node
            scope = util.Scope(nodes[node["base"]]["scope"])
            scope.update(node)
            node["scope"] = scope
        else:
            step['_node'] = node
            scope = util.Scope(default_scopes[steps[0]])
            scope.update(node)
            node["scope"] = scope
#    print_tree(tree)

def print_tree(tree, indent=""):
    """Print statements search tree.
    Intermediate nodes are prefixed with --.
    Useful for debugging.
    """
    parts = tree.get('_key', 'root').split('_')
    if "_node" in tree:
        #        final = '' # + tree["_node"]["scope"].name + '-'
        print("{}{} -- {}".format(indent, parts[-1], tree.get('_key', '??')))
    else:
        print("{}{}".format(indent, parts[-1]))
    indent += '  '
    for key in sorted(tree.keys()):
        if key == '_node':
            continue
        if key == 'scope':
            continue
        if key == '_key':
            continue
        value = tree[key]
        if isinstance(value, dict):
            print_tree(value, indent)

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
    found = default_scopes[path[0]]
    work = []
    step = tree
    for part in path:
        if not part:
            # skip empty parts
            continue
        if part not in step:
            continue
        step = step[part]
        if "_node" in step:
            # Path ends here.
            found = step["_node"]["scope"]
#    if not isinstance(found, util.Scope):
#        raise RuntimeError
    return found


class CStmts(object):
    """C Statements.
    arg_call    - List of arguments passed to C function.

    Used with buf_args = "arg_decl".
    c_arg_decl  - Add C declaration to C wrapper with buf_args=arg_decl
    f_arg_decl  - Add Fortran declaration to Fortran wrapper interface block
                  with buf_args=arg_decl.
    f_result_decl - Declaration for function result.
    f_module    - Add module info to interface block.
    """
    def __init__(self,
        name="c_default",
        buf_args=[], buf_extra=[],
        c_header=[], c_helper="", c_local_var=None,
        cxx_header=[], cxx_local_var=None,
        arg_call=[],
        pre_call=[], call=[], post_call=[], final=[], ret=[],
        destructor_name=None,
        owner="library",
        return_type=None, return_cptr=False,
        c_arg_decl=[],
        f_arg_decl=[],
        f_result_decl=[],
        f_module=None,
    ):
        self.name = name
        self.buf_args = buf_args
        self.buf_extra = buf_extra
        self.c_header = c_header
        self.c_helper = c_helper
        self.c_local_var = c_local_var
        self.cxx_header = cxx_header
        self.cxx_local_var = cxx_local_var

        self.pre_call = pre_call
        self.call = call
        self.arg_call = arg_call
        self.post_call = post_call
        self.final = final
        self.ret = ret

        self.destructor_name = destructor_name
        self.owner = owner
        self.return_type = return_type
        self.return_cptr = return_cptr
        self.c_arg_decl = c_arg_decl
        self.f_arg_decl = f_arg_decl
        self.f_result_decl = f_result_decl
        self.f_module = f_module

class FStmts(object):
    """Fortran Statements.

    """
    def __init__(self,
        name="f_default",
        c_helper="",
        c_local_var=None,
        f_helper="", f_module=None,
        need_wrapper=False,
        arg_name=None,
        arg_decl=None,
        arg_c_call=None,
        declare=[], pre_call=[], call=[], post_call=[],
        result=None,  # name of result variable
    ):
        self.name = name
        self.c_helper = c_helper
        self.c_local_var = c_local_var
        self.f_helper = f_helper
        self.f_module = f_module

        self.need_wrapper = need_wrapper
        self.arg_name = arg_name        # Names in subprogram list.
        self.arg_decl = arg_decl        # argument/result declaration
        self.arg_c_call = arg_c_call    # argument to C function.
        self.declare = declare          # local declaration
        self.pre_call = pre_call
        self.call = call
        self.post_call = post_call
        self.result = result


default_stmts = dict(
    c=CStmts,
    f=FStmts,
)
                
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
# deref      "allocatable", "pointer"

fc_statements = [
    dict(
        name="f_bool_in",
        c_local_var=True,
        pre_call=["{c_var} = {f_var}  ! coerce to C_BOOL"],
    ),
    dict(
        name="f_bool_out",
        c_local_var=True,
        post_call=["{f_var} = {c_var}  ! coerce to logical"],
    ),
    dict(
        name="f_bool_inout",
        c_local_var=True,
        pre_call=["{c_var} = {f_var}  ! coerce to C_BOOL"],
        post_call=["{f_var} = {c_var}  ! coerce to logical"],
    ),
    dict(
        name="f_bool_result",
        # The wrapper is needed to convert bool to logical
        need_wrapper=True
    ),

    dict(
        # A C function with a 'int *' argument passes address of array
        name="f_native_*_in_raw",
        # same as "f_void_*",
        arg_decl=[
            "{f_type}, intent({f_intent}), target :: {f_var}{f_assumed_shape}",
        ],
        f_module=dict(iso_c_binding=["C_LOC"]),
        arg_c_call=["C_LOC({f_var})"],
    ),

    dict(
        # double * out +intent(out) +deref(allocatable)+dimension(size(in)),
        # Allocate array then pass to C wrapper.
        name="f_native_*_out_allocatable",
        arg_decl=[
            "{f_type}, intent({f_intent}), allocatable :: {f_var}{f_assumed_shape}",
        ],
        pre_call=[
            "allocate({f_var}{f_array_allocate})",
        ],
    ),
    
    dict(
        # Any array of pointers.  Assumed to be non-contiguous memory.
        # All Fortran can do is treat as a type(C_PTR).
        name="c_native_**_in",
        buf_args=["arg_decl"],
        c_arg_decl=[
            "{cxx_type} **{cxx_var}",
        ],
        f_arg_decl=[
            "type(C_PTR), intent(IN), value :: {c_var}",
        ],
        f_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        # double **count _intent(out)+dimension(ncount)
        name="c_native_**_out_buf",
        buf_args=["context"],
        c_helper="ShroudTypeDefines",
        pre_call=[
            "{c_const}{cxx_type} *{cxx_var};",
        ],
        arg_call=["&{cxx_var}"],
        post_call=[
            "{c_var_context}->cxx.addr  = {cxx_nonconst_ptr};",
            "{c_var_context}->cxx.idtor = {idtor};",
            "{c_var_context}->addr.base = {cxx_var};",
            "{c_var_context}->type = {sh_type};",
            "{c_var_context}->elem_len = sizeof({cxx_type});",
            "{c_var_context}->rank = {rank};"
            "{c_array_shape}",
            "{c_var_context}->size = {c_array_size};",
        ],
        # XXX - similar to c_native_*_result_buf
    ),
    dict(
        name="c_native_*&_out_buf",
        base="c_native_**_out_buf",
        arg_call=["{cxx_var}"],
    ),
    dict(
        # deref(pointer)
        # A C function with a 'int **' argument associates it
        # with a Fortran pointer to a scalar.
        name="f_XXX_native_**_out",
        arg_decl=[
            "{f_type}, intent({f_intent}), pointer :: {f_var}",
        ],
        f_module=dict(iso_c_binding=["C_PTR", "c_f_pointer"]),
        declare=[
            "type(C_PTR) :: {F_pointer}",
        ],
        arg_c_call=["{F_pointer}"],
        post_call=[
            "call c_f_pointer({F_pointer}, {f_var})",
        ],
    ),
    dict(
        # deref(pointer)
        # A C function with a 'int **' argument associates it
        # with a Fortran pointer.
        name="f_native_**_out",
        arg_decl=[
            "{f_type}, intent({f_intent}), pointer :: {f_var}{f_assumed_shape}",
        ],
        f_module=dict(iso_c_binding=["c_f_pointer"]),
        post_call=[
            "call c_f_pointer({c_var_context}%base_addr, {f_var}{f_array_shape})",
        ],
    ),
    dict(
        # Make argument type(C_PTR) from 'int **'
        name="f_native_**_out_raw",
        arg_decl=[
            "type(C_PTR), intent({f_intent}) :: {f_var}",
        ],
        declare=[
            "type({F_array_type}) {c_var_context}",
        ],
        f_module=dict(iso_c_binding=["C_PTR"]),
        arg_c_call=["{c_var_context}"],
        # This post_call block will set need_wrapper=True
        # No real need for F_array_type since C_PTR can be passed directly
        # but c_native_**_out_buf uses buf_args=context.
        # XXX - maybe use c_native_**_out_buf_raw
        post_call=[
            "{f_var} = {c_var_context}%base_addr",
        ],
    ),
    dict(
        name="f_native_*&_out",
        base="f_native_**_out",
    ),

    # XXX only in buf?
    # Used with intent IN, INOUT, and OUT.
#    c_native_pointer_cdesc=dict(
    dict(
        name="c_native_*_cdesc",
        buf_args=["context"],
#        c_helper="ShroudTypeDefines",
        c_pre_call=[
            "{cxx_type} * {c_var} = {c_var_context}->addr.base;",
        ],
        cxx_pre_call=[
#            "{cxx_type} * {c_var} = static_cast<{cxx_type} *>\t"
#            "({c_var_context}->addr.base);",
            "{cxx_type} * {c_var} = static_cast<{cxx_type} *>\t"
            "(const_cast<void *>({c_var_context}->addr.base));",
        ],
    ),
#    f_native_pointer_cdesc=dict(
    dict(
        name="f_native_*_cdesc",
        # TARGET required for argument to C_LOC.
        arg_decl=[
            "{f_type}, intent({f_intent}), target :: {f_var}{f_assumed_shape}",
        ],
        f_helper="ShroudTypeDefines",
        f_module=dict(iso_c_binding=["C_LOC"]),
#        initialize=[
        pre_call=[
            "{c_var_context}%base_addr = C_LOC({f_var})",
            "{c_var_context}%type = {sh_type}",
            "! {c_var_context}%elem_len = C_SIZEOF()",
#            "{c_var_context}%size = size({f_var})",
            "{c_var_context}%size = {size}",
            "{c_var_context}%rank = {rank}",
            # This also works with scalars since (1:0) is a zero length array.
            "{c_var_context}%shape(1:{rank}) = shape({f_var})",
        ],
    ),
    dict(
        name="f_native_*_in_cdesc",
        base="f_native_*_cdesc",
    ),
    dict(
        name="f_native_*_out_cdesc",
        base="f_native_*_cdesc",
    ),

########################################
# void *
    dict(
        name="f_void_*_in",
        f_module=dict(iso_c_binding=["C_PTR"]),
        arg_decl=[
            "type(C_PTR), intent(IN) :: {f_var}",
        ],
    ),
    dict(
        # return a type(C_PTR)
        name="f_void_*_result",
        f_module=dict(iso_c_binding=["C_PTR"]),
        arg_decl=[
            "type(C_PTR) :: {f_var}",
        ],
    ),
    dict(
        name="f_void_**_out",
        f_module=dict(iso_c_binding=["C_PTR"]),
        arg_decl=[
            "type(C_PTR), intent(OUT) :: {f_var}",
        ],
    ),
    
    dict(
        name="c_void_*_cdesc",
        base="c_native_*_cdesc",
    ),
    dict(
        name="f_void_*_cdesc",
        base="f_native_*_cdesc",
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
    #
    #        c_step1(context)
    #        call c_f_pointer(c_ptr, f_ptr, shape)
    dict(
        name="c_native_*_result_buf",
        buf_args=["context"],
        c_helper="ShroudTypeDefines",
        post_call=[
            "{c_var_context}->cxx.addr  = {cxx_nonconst_ptr};",
            "{c_var_context}->cxx.idtor = {idtor};",
            "{c_var_context}->addr.base = {cxx_var};",
            "{c_var_context}->type = {sh_type};",
            "{c_var_context}->elem_len = sizeof({cxx_type});",
            "{c_var_context}->rank = {rank};"
            "{c_array_shape}",
            "{c_var_context}->size = {c_array_size};",
        ],
        return_cptr=True,
    ),
    dict(
        name="f_native_*_result_allocatable",
        c_helper="copy_array",
        f_helper="copy_array_{cxx_type}",
        f_module=dict(iso_c_binding=["C_PTR"]),
        declare=[
            "type(C_PTR) :: {F_pointer}",
        ],
        call=[
            "{F_pointer} = {F_C_call}({F_arg_c_call})",
        ],
        post_call=[
            # XXX - allocate scalar
            "allocate({f_var}({c_var_dimension}))",
            "call SHROUD_copy_array_{cxx_type}"
            "({c_var_context}, {f_var}, size({f_var}, kind=C_SIZE_T))",
        ],
    ),

    # f_pointer_shape may be blank for a scalar, otherwise it
    # includes a leading comma.
    dict(
        name="f_native_*_result_pointer",
        f_module=dict(iso_c_binding=["C_PTR", "c_f_pointer"]),
        declare=[
            "type(C_PTR) :: {F_pointer}",
        ],
        call=[
            "{F_pointer} = {F_C_call}({F_arg_c_call})",
        ],
        post_call=[
            "call c_f_pointer({F_pointer}, {F_result}{f_array_shape})",
        ],
    ),
    dict(
        # +deref(pointer) +owner(caller)
        name="f_native_*_result_pointer_caller",
        f_helper="capsule_helper",
        f_module=dict(iso_c_binding=["C_PTR", "c_f_pointer"]),
        arg_name=["{c_var_capsule}"],
        arg_decl=[
            "{f_type}, pointer :: {f_var}{f_assumed_shape}",
            "type({F_capsule_type}), intent(OUT) :: {c_var_capsule}",
        ],
        declare=[
            "type(C_PTR) :: {F_pointer}",
        ],
        call=[
            "{F_pointer} = {F_C_call}({F_arg_c_call})",
        ],
        post_call=[
            "call c_f_pointer({F_pointer}, {F_result}{f_array_shape})",
            "{c_var_capsule}%mem = {c_var_context}%cxx",
        ],
    ),
    dict(
        name="f_native_*_result_raw",
        arg_decl=[
            "type(C_PTR) :: {f_var}",
        ],
    ),
    dict(
        # int **func(void)
        # regardless of deref value.
        name="f_native_**_result",
        arg_decl=[
            "type(C_PTR) :: {f_var}",
        ],
    ),
    
    dict(
        name="f_native_&_result",
        base="f_native_*_result_pointer",   # XXX - change base to &?
    ),

    dict(
        name="f_native_*_result_scalar",
        # avoid catching f_native_*_result
    ),


    ########################################
    # char arg
    dict(
        name="c_char_scalar_in",
        buf_args=["arg_decl"],
        c_arg_decl=[
            "char {c_var}",
        ],
        f_arg_decl=[
            "character(kind=C_CHAR), value, intent(IN) :: {c_var}",
        ],
        f_module=dict(iso_c_binding=["C_CHAR"]),
    ),
    dict(
        name="f_char_scalar_in",
        # By default the declaration is character(LEN=*).
        arg_decl=[
            "character, value, intent(IN) :: {f_var}",
        ],
    ),
    dict(
        name="c_char_scalar_result",
        f_result_decl=[
            "character(kind=C_CHAR) :: {c_var}",
        ],
        f_module=dict(iso_c_binding=["C_CHAR"]),
    ),
    dict(
        name="c_char_scalar_result_buf",
        buf_args=["arg", "len"],
        c_header=["<string.h>"],
        cxx_header=["<cstring>"],
        post_call=[
            "{stdlib}memset({c_var}, ' ', {c_var_len});",
            "{c_var}[0] = {cxx_var};",
        ],
    ),
    
    dict(
        name="c_char_*_result",
        return_cptr=True,
    ),
    dict(
        name="c_char_*_in_buf",
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
    dict(
        name="c_char_*_out_buf",
        buf_args=["arg", "len"],
        c_helper="ShroudStrBlankFill",
        post_call=[
            "ShroudStrBlankFill({c_var}, {c_var_len});"
        ],
    ),
    dict(
        name="c_char_*_inout_buf",
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
    dict(
        name="c_char_*_result_buf",
        buf_args=["arg", "len"],
        c_helper="ShroudStrCopy",
        post_call=[
            # nsrc=-1 will call strlen({cxx_var})
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {cxx_var},\t -1);",
        ],
    ),
    dict(
        name="c_char_*_result_buf_allocatable",
        buf_args=["context"],
        c_helper="ShroudTypeDefines",
        # Copy address of result into c_var and save length.
        # When returning a std::string (and not a reference or pointer)
        # an intermediate object is created to save the results
        # which will be passed to copy_string
        post_call=[
            "{c_var_context}->cxx.addr = {cxx_nonconst_ptr};",
            "{c_var_context}->cxx.idtor = {idtor};",
            "{c_var_context}->addr.ccharp = {cxx_var};",
            "{c_var_context}->type = {sh_type};",
            "{c_var_context}->elem_len = {cxx_var} == {nullptr} ? 0 : {stdlib}strlen({cxx_var});",
            "{c_var_context}->size = 1;",
            "{c_var_context}->rank = 0;",
        ],
    ),

    dict(
        # char *func() +deref(raw)
        name="f_char_*_result_raw",
        arg_decl=[
            "type(C_PTR) :: {f_var}",
        ],
    ),
    #####
    dict(
        # Treat as an assumed length array in Fortran interface.
        name='c_char_**_in',
        buf_args=["arg_decl"],
        c_arg_decl=[
            "char **{c_var}",
        ],
        f_arg_decl=[
            "type(C_PTR), intent(IN) :: {c_var}(*)",
        ],
        f_module=dict(iso_c_binding=["C_PTR"]),
    ),
    dict(
        name='c_char_**_in_buf',
        # arg_decl - argument is char *, not char **.
        buf_args=["arg_decl", "size", "len"],
        c_helper="ShroudStrArrayAlloc ShroudStrArrayFree",
        cxx_local_var="pointer",
        pre_call=[
            "char **{cxx_var} = ShroudStrArrayAlloc("
            "{c_var},\t {c_var_size},\t {c_var_len});",
        ],
        post_call=[
            "ShroudStrArrayFree({cxx_var}, {c_var_size});",
        ],

        c_arg_decl=[
            "char *{c_var}",
        ],
        f_arg_decl=[
            "character(kind=C_CHAR), intent(IN) :: {c_var}(*)",
        ],
        f_module=dict(iso_c_binding=["C_CHAR"]),
    ),
    #####
    dict(
        name="f_char_*_result_allocatable",
        need_wrapper=True,
        c_helper="copy_string",
        f_helper="copy_string",
        arg_decl=[
            "character(len=:), allocatable :: {f_var}",
        ],
        post_call=[
            "allocate(character(len={c_var_context}%elem_len):: {f_var})",
            "call SHROUD_copy_string_and_free"
            "({c_var_context}, {f_var}, {c_var_context}%elem_len)",
        ],
    ),
    dict(
        name="f_char_scalar_result_allocatable",
        base="f_char_*_result_allocatable",
    ),

    dict(
        name="c_string_in",
        cxx_local_var="scalar",
        pre_call=["{c_const}std::string {cxx_var}({c_var});"],
    ),
    dict(
        name="c_string_out",
        cxx_header=["<cstring>"],
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
    dict(
        name="c_string_inout",
        cxx_header=["<cstring>"],
        cxx_local_var="scalar",
        pre_call=["{c_const}std::string {cxx_var}({c_var});"],
        post_call=[
            # This may overwrite c_var if cxx_val is too long
            "strcpy({c_var}, {cxx_var}{cxx_member}c_str());"
        ],
    ),
    dict(
        name="c_string_in_buf",
        buf_args=["arg", "len_trim"],
        cxx_local_var="scalar",
        pre_call=[
            (
                "{c_const}std::string "
                "{cxx_var}({c_var}, {c_var_trim});"
            )
        ],
    ),
    dict(
        name="c_string_out_buf",
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
    dict(
        name="c_string_inout_buf",
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
    dict(
        name="c_string_result",
        # cxx_to_c creates a pointer from a value via c_str()
        # The default behavior will dereference the value.
        ret=[
            "return {c_var};",
        ],
        return_cptr=True,
    ),
    dict(
        name="c_string_result_buf",
        buf_args=["arg", "len"],
        c_helper="ShroudStrCopy",
        post_call=[
            "if ({cxx_var}{cxx_member}empty()) {{+",
            "ShroudStrCopy({c_var}, {c_var_len},"
            "\t {nullptr},\t 0);",
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
    #    allocate(character(len=context%elem_len): Fout)
    #    c_step2(context, Fout, context%elem_len)
    # only used with bufferifed routines and intent(out) or result
    # std::string * function()
    dict(
        name="c_string_result_buf_allocatable",
        # pass address of string and length back to Fortran
        buf_args=["context"],
        c_helper="ShroudStrToArray",
        # Copy address of result into c_var and save length.
        # When returning a std::string (and not a reference or pointer)
        # an intermediate object is created to save the results
        # which will be passed to copy_string
        post_call=[
            "ShroudStrToArray({c_var_context}, {cxx_addr}{cxx_var}, {idtor});",
        ],
    ),

    # Since 'c_string_scalar_result_buf_allocatable' exists,
    # must set an alias for c_string_scalar.
    # No need to allocate a local copy since the string is copied
    # into a Fortran variable before the string is deleted.
    dict(
        name="c_string_scalar_result_buf",
        base="c_string_result_buf",
    ),
    
    # std::string function()
    # Must allocate the std::string then assign to it via cxx_rv_decl.
    # This allows the std::string to outlast the function return.
    # The Fortran wrapper will ALLOCATE memory, copy then delete the string.
    dict(
        name="c_string_scalar_result_buf_allocatable",
        # pass address of string and length back to Fortran
        buf_args=["context"],
        cxx_local_var="pointer",
        c_helper="ShroudStrToArray",
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
    
    # similar to f_char_result_allocatable
    dict(
        name="f_string_result_allocatable",
        need_wrapper=True,
        c_helper="copy_string",
        f_helper="copy_string",
        arg_decl=[
            "character(len=:), allocatable :: {f_var}",
        ],
        post_call=[
            "allocate(character(len={c_var_context}%elem_len):: {f_var})",
            "call SHROUD_copy_string_and_free("
            "{c_var_context}, {f_var}, {c_var_context}%elem_len)",
        ],
    ),
    
    
    dict(
        name="c_vector_in_buf",
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
    dict(
        name="c_vector_out_buf",
        buf_args=["context"],
        cxx_local_var="pointer",
        c_helper="ShroudTypeDefines",
        pre_call=[
            "{c_const}std::vector<{cxx_T}>"
            "\t *{cxx_var} = new std::vector<{cxx_T}>;"
        ],
        post_call=[
            # Return address and size of vector data.
            "{c_var_context}->cxx.addr  = {cxx_var};",
            "{c_var_context}->cxx.idtor = {idtor};",
            "{c_var_context}->addr.base = {cxx_var}->empty()"
            " ? {nullptr} : &{cxx_var}->front();",
            "{c_var_context}->type = {sh_type};",
            "{c_var_context}->elem_len = sizeof({cxx_T});",
            "{c_var_context}->size = {cxx_var}->size();",
            "{c_var_context}->rank = 1;",
            "{c_var_context}->shape[0] = {c_var_context}->size;",
        ],
        destructor_name="std_vector_{cxx_T}",
        destructor=[
            "std::vector<{cxx_T}> *cxx_ptr ="
            " \treinterpret_cast<std::vector<{cxx_T}> *>(ptr);",
            "delete cxx_ptr;",
        ],
    ),
    dict(
        name="c_vector_inout_buf",
        buf_args=["arg", "size", "context"],
        cxx_local_var="pointer",
        c_helper="ShroudTypeDefines",
        pre_call=[
            "std::vector<{cxx_T}> *{cxx_var} = \tnew std::vector<{cxx_T}>\t("
            "\t{c_var}, {c_var} + {c_var_size});"
        ],
        post_call=[
            # Return address and size of vector data.
            "{c_var_context}->cxx.addr  = {cxx_var};",
            "{c_var_context}->cxx.idtor = {idtor};",
            "{c_var_context}->addr.base = {cxx_var}->empty()"
            " ? {nullptr} : &{cxx_var}->front();",
            "{c_var_context}->type = {sh_type};",
            "{c_var_context}->elem_len = sizeof({cxx_T});",
            "{c_var_context}->size = {cxx_var}->size();",
            "{c_var_context}->rank = 1;",
            "{c_var_context}->shape[0] = {c_var_context}->size;",
        ],
        destructor_name="std_vector_{cxx_T}",
        destructor=[
            "std::vector<{cxx_T}> *cxx_ptr ="
            " \treinterpret_cast<std::vector<{cxx_T}> *>(ptr);",
            "delete cxx_ptr;",
        ],
    ),
    # Almost same as intent_out_buf.
    dict(
        name="c_vector_result_buf",
        buf_args=["context"],
        cxx_local_var="pointer",
        c_helper="ShroudTypeDefines",
        pre_call=[
            "{c_const}std::vector<{cxx_T}>"
            "\t *{cxx_var} = new std::vector<{cxx_T}>;"
        ],
        post_call=[
            # Return address and size of vector data.
            "{c_var_context}->cxx.addr  = {cxx_var};",
            "{c_var_context}->cxx.idtor = {idtor};",
            "{c_var_context}->addr.base = {cxx_var}->empty()"
            " ? {nullptr} : &{cxx_var}->front();",
            "{c_var_context}->type = {sh_type};",
            "{c_var_context}->elem_len = sizeof({cxx_T});",
            "{c_var_context}->size = {cxx_var}->size();",
            "{c_var_context}->rank = 1;",
            "{c_var_context}->shape[0] = {c_var_context}->size;",
        ],
        destructor_name="std_vector_{cxx_T}",
        destructor=[
            "std::vector<{cxx_T}> *cxx_ptr ="
            " \treinterpret_cast<std::vector<{cxx_T}> *>(ptr);",
            "delete cxx_ptr;",
        ],
    ),
    #                dict(
    #                    name="c_vector_result_buf",
    #                    buf_args=['arg', 'size'],
    #                    c_helper='ShroudStrCopy',
    #                    post_call=[
    #                        'if ({cxx_var}.empty()) {{+',
    #                        'ShroudStrCopy({c_var}, {c_var_len},'
    #                        '{nullptr}, 0);',
    #                        '-}} else {{+',
    #                        'ShroudStrCopy({c_var}, {c_var_len},'
    #                        '\t {cxx_var}{cxx_member}data(),'
    #                        '\t {cxx_var}{cxx_member}size());',
    #                        '-}}',
    #                    ],
    #                ),
    
    # Specialize for vector<string>.
    dict(
        name="c_vector_in_buf_string",
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
    dict(
        name="c_vector_out_buf_string",
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
    dict(
        name="c_vector_inout_buf_string",
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
    #                    dict(
    #                        name="c_vector_result_buf_string",
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
    dict(
        name="f_vector_out",
        c_helper="copy_array",
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "call SHROUD_copy_array_{cxx_T}({c_var_context}, "
            "{f_var}, size({f_var},kind=C_SIZE_T))"
        ],
    ),
    dict(
        name="f_vector_inout",
        c_helper="copy_array",
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "call SHROUD_copy_array_{cxx_T}({c_var_context}, "
            "{f_var}, size({f_var},kind=C_SIZE_T))"
        ],
    ),
    dict(
        name="f_vector_result",
        c_helper="copy_array",
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "call SHROUD_copy_array_{cxx_T}({c_var_context}, "
            "{f_var}, size({f_var},kind=C_SIZE_T))"
        ],
    ),
    # copy into allocated array
    dict(
        name="f_vector_out_allocatable",
        c_helper="copy_array",
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "allocate({f_var}({c_var_context}%size))",
            "call SHROUD_copy_array_{cxx_T}({c_var_context}, "
            "{f_var}, size({f_var},kind=C_SIZE_T))",
        ],
    ),
    dict(
        name="f_vector_inout_allocatable",
        c_helper="copy_array",
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "if (allocated({f_var})) deallocate({f_var})",
            "allocate({f_var}({c_var_context}%size))",
            "call SHROUD_copy_array_{cxx_T}({c_var_context}, "
            "{f_var}, size({f_var},kind=C_SIZE_T))",
        ],
    ),
    # Similar to f_vector_out_allocatable but must declare result variable.
    # Always return a 1-d array.
    dict(
        name="f_vector_result_allocatable",
        c_helper="copy_array",
        f_helper="copy_array_{cxx_T}",
        f_module=dict(iso_c_binding=["C_SIZE_T"]),
        post_call=[
            "allocate({f_var}({c_var_context}%size))",
            "call SHROUD_copy_array_{cxx_T}({c_var_context}, "
            "{f_var}, size({f_var},kind=C_SIZE_T))",
        ],
    ),

    # Pass in a pointer to a shadow object via buf_args.
    # Extract pointer to C++ instance.
    # convert C argument into a pointer to C++ type.
    dict(
        name="c_shadow_in",
        buf_args=["shadow"],
        cxx_local_var="pointer",
        pre_call=[
            "{c_const}{cxx_type} * {cxx_var} =\t "
            "static_cast<{c_const}{cxx_type} *>\t({c_var}{c_member}addr);",
        ],
    ),
    dict(
        name="c_shadow_inout",
        base="c_shadow_in",
    ),
    dict(
        name="c_shadow_scalar_in",
        base="c_shadow_in",
    ),
    # Return a C_capsule_data_type.
    dict(
        name="c_shadow_result",
        buf_extra=["shadow"],
        c_local_var="pointer",
        post_call=[
            "{c_var}->addr = {cxx_nonconst_ptr};",
            "{c_var}->idtor = {idtor};",
        ],
        ret=[
            "return {c_var};",
        ],
        return_type="{c_type} *",
        return_cptr=True,
    ),
    dict(
        name="c_shadow_scalar_result",
        # Return a instance by value.
        # Create memory in pre_call so it will survive the return.
        # owner="caller" sets idtor flag to release the memory.
        # c_local_var is passed in via buf_extra=shadow.
        buf_extra=["shadow"],
        cxx_local_var="pointer",
        c_local_var="pointer",
        owner="caller",
        pre_call=[
            "{cxx_type} * {cxx_var} = new {cxx_type};",
        ],
        post_call=[
            "{c_var}->addr = {cxx_nonconst_ptr};",
            "{c_var}->idtor = {idtor};",
        ],
        ret=[
            "return {c_var};",
        ],
        return_type="{c_type} *",
        return_cptr=True,
    ),
    dict(
        name="f_shadow_result",
        need_wrapper=True,
        f_module=dict(iso_c_binding=["C_PTR"]),
        declare=[
            "type(C_PTR) :: {F_result_ptr}",
        ],
        call=[
            # The C function returns a pointer.
            # Save in a type(C_PTR) variable.
            "{F_result_ptr} = {F_C_call}({F_arg_c_call})"
        ],
    ),
    dict(
        name="c_shadow_ctor",
        buf_extra=["shadow"],
        cxx_local_var="pointer",
        call=[
            "{cxx_type} *{cxx_var} =\t new {cxx_type}({C_call_list});",
            "{c_var}->addr = static_cast<{c_const}void *>(\t{cxx_var});",
            "{c_var}->idtor = {idtor};",
        ],
        ret=[
            "return {c_var};",
        ],
        return_type="{c_type} *",
        owner="caller",
    ),
    dict(
        name="c_shadow_scalar_ctor",
        base="c_shadow_ctor",
    ),
    dict(
        name="f_shadow_ctor",
        base="f_shadow_result",
    ),
    dict(
        # NULL in stddef.h
        name="c_shadow_dtor",
        c_header=["<stddef.h>"],
        cxx_header=["<cstddef>"],
        call=[
            "delete {CXX_this};",
            "{C_this}->addr = {nullptr};",
        ],
        return_type="void",
    ),

    dict(
        # Used with in, out, inout
        # C pointer -> void pointer -> C++ pointer
        name="c_struct",
        cxx_cxx_local_var="pointer", # cxx_local_var only used with C++
        cxx_pre_call=[
            "{c_const}{cxx_type} * {cxx_var} = \tstatic_cast<{c_const}{cxx_type} *>\t(static_cast<{c_const}void *>(\t{c_addr}{c_var}));",
        ],
    ),
    dict(
        name="c_struct_result",
        # C++ pointer -> void pointer -> C pointer
        c_local_var="pointer",
        cxx_post_call=[
            "{c_const}{c_type} * {c_var} = \tstatic_cast<{c_const}{c_type} *>(\tstatic_cast<{c_const}void *>(\t{cxx_addr}{cxx_var}));",
        ],
    ),
    dict(
        name="f_struct_scalar_result",
        # Needed to differentiate from f_struct_pointer_result.
    ),
    dict(
        name="f_struct_*_result",
        base="f_native_*_result_pointer",
    ),
]
