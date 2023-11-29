#!/bin/env python3
# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
########################################################################
"""
generate language bindings
"""

from __future__ import print_function
from __future__ import absolute_import

import argparse
import json
import os
import sys
import yaml
from yaml.composer import Composer
from yaml.constructor import Constructor

from . import ast
from . import declast
from . import fcfmt
from . import generate
from . import metaattrs
from . import metadata
from . import splicer
from . import statements
from . import todict
from . import typemap
from . import util
from . import whelpers
from . import wrapc
from . import wrapf
from . import wrapp
from . import wrapl


class Config(object):
    """A class to stash configuration values.
    """

    def __init__(self):
        self.c_fortran_dir = ""
        self.python_dir = ""
        self.lua_dir = ""
        self.out_dir = ""
        self.yaml_dir = ""
        self.yaml_types = ""
        self.log = ""
        self.cfiles = []  # list of C/C++ files created
        self.ffiles = []  # list of Fortran files created
        self.pyfiles = []  # list of Python module files created
        self.fc_shared_helpers = {}   # Shared between Fortran and C.


class TypeOut(util.WrapperMixin):
    """A class to write out Class type information.
    It may be 'imported' by another file to share classes across YAML files.
    It subclasses util.WrapperMixin in order to access
    write routines.
    """

    def __init__(self, newlibrary, config):
        self.newlibrary = newlibrary
        self.config = config
        self.log = config.log
        self.comment = "#"
        self.cont = ""
        self.linelen = 1000

    def _get_namespaces(self, node, top):
        for cls in node.classes:
            fullname = cls.typemap.name
            parts = fullname.split("::")
            top[parts[-1]] = cls.typemap
        for ns in node.namespaces:
            top[ns.name] = {}
            self._get_namespaces(ns, top[ns.name])
            
    def splitup(self, ns, output, mode):
        for name in sorted(ns.keys()):
            nxt = ns[name]
            if isinstance(nxt, dict):
                if nxt:
                    output.append("@- namespace: " + name)
                    output.append(1)
                    output.append("declarations: " + name)
                    self.splitup(nxt, output, mode)
                    output.append(-1)
            elif isinstance(nxt, typemap.Typemap):
                output.append("@- type: " + name)
                output.append(1)
                output.append("fields:")
                output.append(1)
                nxt.__export_yaml__(output, mode)
                output.append(-2)
            else:
                raise RuntimeError("Unexpected class in splitup")

    def write_class_types(self):
        """Write out types into a file.
        This file can be read by Shroud to share types.
        """
        newlibrary = self.newlibrary
        newlibrary.eval_template("YAML_type_filename")
        fname = newlibrary.fmtdict.YAML_type_filename

        top = {}
        #        get_namespaces(newlibrary.wrap_namespace, top)
        self._get_namespaces(newlibrary, top)
        # Write out typemaps from YAML file.
        for key, ntypemap in newlibrary.symtab.typemaps.items():
            if ntypemap.export:
                top[key] = ntypemap

        output = []
        self.splitup(top, output, "class")

        if output:
            output = [
                "# Types generated by Shroud for library '{}'".format(
                    self.newlibrary.library
                ),
                "typemap:",
                1,
            ] + output

            self.write_output_file(
                fname, self.config.yaml_dir, output, spaces="  "
            )

    def write_all_types(self, def_types, fname):
        """Write out types into a file.
        This file can be read by Shroud to share types.
        """
        output = []
        self.splitup(def_types, output, "all")

        # This is called before the YAML file is read and the library created.
        # Dummy out the library for the copyright.
        class Dummy:
            copyright = ["""
# Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
# other Shroud Project Developers.
# See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)"""]
        self.newlibrary = Dummy()

        if output:
            output = [
                "# All types predefined by Shroud",
                "typemap:",
                1,
            ] + output

            self.write_output_file(
                fname, self.config.yaml_dir, output, spaces="  "
            )

def prune_entries(dct, names=[]):
    """Recursively remove names from dct."""
    for key in names:
        if key in dct:
            del dct[key]
    for key, value in dct.items():
        if isinstance(value, dict):
            prune_entries(value, names)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    prune_entries(item, names)

def dump_jsonfile(config, logdir, basename, newlibrary):
    """Write a JSON file for debugging.
    """
    jsonpath = os.path.join(logdir, basename + ".json")
    fp = open(jsonpath, "w")

    out = dict(
        library=todict.to_dict(newlibrary),
        # yaml=all,
    )
    # This notice should sort to the top.
    out["<NOTICE>"] = "This file is generated by Shroud {} " \
                      "and is useful for debugging.".format(config.write_version)

    typemaps = newlibrary.symtab.typemaps
    if config.yaml_types:
        # Dump types if requested.
        out["types"] = todict.to_dict(typemaps)
    elif newlibrary.options.debug_testsuite:
        # Add user defined types for debugging.
        user_types = typemap.return_shadow_types(typemaps)
        if user_types:
            out["types"] = todict.to_dict(user_types)
        user_types = declast.symtab_to_typemap(newlibrary.symtab.scope_stack[0])
        if user_types:
            out["symtab"] = todict.to_dict(user_types)
        # Clean out this info since it's the same for all tests.
        # XXX - anytime a new fmt or option is added it changes all tests.
        del out['library']['zz_fmtdict']
        del out['library']['options']
        prune_entries(out, names=['__line__', 'linenumber'])

    json.dump(out, fp, sort_keys=True, indent=4, separators=(',', ': '))
    fp.close()

def dump_fmt(config, logdir, basename, newlibrary):
    """Write a JSON file for debugging.
    """
    jsonpath = os.path.join(logdir, basename + ".fmt.json")
    fp = open(jsonpath, "w")

    out = dict(
        library=todict.print_fmt(newlibrary),
        # yaml=all,
    )

    json.dump(out, fp, sort_keys=True, indent=4, separators=(',', ': '))
    fp.close()

    
# https://thisdataguy.com/2017/07/03/no-options-with-argparse-and-python/
class BooleanAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(BooleanAction, self).__init__(
            option_strings, dest, nargs=0, **kwargs)
 
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest,
                False if option_string.startswith('--no') else True)


def main():
    appname = "shroud"

    parser = argparse.ArgumentParser(
        prog=appname,
        description="""Create Fortran or Python wrapper for a C++ library.
""",
    )
    parser.add_argument("--version", action="version",
                        version=metadata.__version__)
    parser.add_argument(
        "--outdir",
        default="",
        help="Directory for output files." "Defaults to current directory.",
    )
    parser.add_argument(
        "--outdir-c-fortran",
        default="",
        dest="outdir_c_fortran",
        help="Directory for C/Fortran wrapper output files, "
        "overrides --outdir.",
    )
    parser.add_argument(
        "--outdir-python",
        default="",
        dest="outdir_python",
        help="Directory for Python wrapper output files, "
        "overrides --outdir.",
    )
    parser.add_argument(
        "--outdir-lua",
        default="",
        dest="outdir_lua",
        help="Directory for Lua wrapper output files, " "overrides --outdir.",
    )
    parser.add_argument(
        "--outdir-yaml",
        default="",
        dest="outdir_yaml",
        help="Directory for yaml output files, " "overrides --outdir.",
    )
    parser.add_argument(
        "--logdir",
        default="",
        help="Directory for log files." "Defaults to current directory.",
    )
    parser.add_argument(
        "--cfiles",
        default="",
        help="Output file with list of C and C++ files " "created.",
    )
    parser.add_argument(
        "--ffiles", default="", help="Output file with list of Fortran created"
    )
    parser.add_argument(
        "--path",
        default=[],
        action="append",
        help="Colon delimited paths to search for "
        "splicer files, may be supplied multiple "
        "times to append to path.",
    )
    parser.add_argument(
        "--cmake", default="", help="Create a file with CMake macro"
    )
    parser.add_argument(
        "--write-helpers", default="", help="Write a file with helper functions."
    )
    parser.add_argument(
        "--write-statements", default="", help="Write a file with statements."
    )
    parser.add_argument(
        "--write-version", "--nowrite-version",
        dest="write_version",
        action=BooleanAction,
        default=True,
        help="Write version into generated files. --nowrite-version to not write version"
    )
    parser.add_argument(
        "--yaml-types", default="", help="Write a YAML file with default types."
    )
    parser.add_argument("filename", nargs="*", help="Input file to process.")

    parser.add_argument("--option", default=[], action="append",
                        help="Define an option with value")
    parser.add_argument("--language", choices=['c', 'c++'],
                        help="Input language.")

    args = parser.parse_args()
    main_with_args(args)
    #    sys.stderr.write("Some useful message")  # example error message
    sys.exit(0)  # set status for errors


def create_wrapper(filename, outdir="", path=None):
    """Translate function arguments into command line options.
    Return config instance. It has list of files created.
    Create a wrapper from a Python program.
    Useful with setup.py.
       config = shroud.create_wrapper('../../ownership.yaml', outdir=outdir)
    """
    args = argparse.Namespace()
    args.cmake = ""
    args.cfiles = ""
    args.ffiles = ""
    args.filename = [filename]
    args.logdir = ""
    args.outdir = outdir
    args.outdir_c_fortran = ""
    args.outdir_lua = ""
    args.outdir_python = ""
    args.outdir_yaml = ""
    if path is None:
        args.path = []
    else:
        args.path = path
    args.write_helpers = ""
    args.write_statements = ""
    args.yaml_types = ""

    config = main_with_args(args)

    return config


def main_with_args(args):
    """Main after args have been parsed.
    Useful for testing.
    """
    if args.cmake:
        # Create C make file
        try:
            fp = open(args.cmake, "w")
            fp.write(whelpers.cmake)
            fp.close()
            raise SystemExit
        except IOError as e:
            print(str(e))
            raise SystemExit(1)

    # check command line options
    if len(args.filename) == 0:
        raise SystemExit("Must give at least one input file")
    if args.outdir and not os.path.isdir(args.outdir):
        raise SystemExit("outdir {} does not exist".format(args.outdir))
    if args.outdir_c_fortran and not os.path.isdir(args.outdir_c_fortran):
        raise SystemExit(
            "outdir-fortran {} does not exist".format(args.outdir_c_fortran)
        )
    if args.outdir_python and not os.path.isdir(args.outdir_python):
        raise SystemExit(
            "outdir-python {} does not exist".format(args.outdir_python)
        )
    if args.outdir_lua and not os.path.isdir(args.outdir_lua):
        raise SystemExit("outdir-lua {} does not exist".format(args.outdir_lua))
    if args.outdir_yaml and not os.path.isdir(args.outdir_yaml):
        raise SystemExit(
            "outdir-yaml {} does not exist".format(args.outdir_yaml)
        )
    if args.logdir and not os.path.isdir(args.logdir):
        raise SystemExit("logdir {} does not exist".format(args.logdir))

    # append all paths together
    if args.path:
        search_path = []
        for pth in args.path:
            search_path.extend(pth.split(":"))
    else:
        search_path = ["."]

    basename = os.path.splitext(os.path.basename(args.filename[0]))[0]
    logpath = os.path.join(args.logdir, basename + ".log")
    log = open(logpath, "w")

    # pass around a configuration object
    config = Config()
    config.out_dir = args.outdir
    config.c_fortran_dir = args.outdir_c_fortran or args.outdir
    config.python_dir = args.outdir_python or args.outdir
    config.lua_dir = args.outdir_lua or args.outdir
    config.yaml_dir = args.outdir_yaml or args.outdir
    config.write_helpers = args.write_helpers
    config.write_statements = args.write_statements
    config.yaml_types = args.yaml_types
    config.log = log
    if args.write_version:
        config.write_version = metadata.__version__
    else:
        config.write_version = "nowrite-version"
    #    config.cfiles = []  # list of C/C++ files created
    #    config.ffiles = []  # list of Fortran files created
    #    config.pyfiles = [] # list of Python module files created

    # accumulated input
    allinput = {}
    splicers = dict(c={}, f={}, py={}, lua={})

    for filename in args.filename:
        ext = os.path.splitext(filename)[1]
        if ext in [".yaml", ".yml"]:
            # print("Read %s" % os.path.basename(filename))
            log.write("Read yaml %s\n" % os.path.basename(filename))
            fp = open(filename, "r")
            # d = yaml.load(fp.read())  # no line numbers

            # https://stackoverflow.com/questions/13319067/
            # parsing-yaml-return-with-line-number
            loader = yaml.Loader(fp.read())

            def compose_node(parent, index):
                # the line number where the previous token has ended (plus empty lines)
                line = loader.line
                node = Composer.compose_node(loader, parent, index)
                node.__line__ = line + 1
                return node

            def construct_mapping(node, deep=False):
                mapping = Constructor.construct_mapping(loader, node, deep=deep)
                mapping["__line__"] = node.__line__
                return mapping

            loader.compose_node = compose_node
            loader.construct_mapping = construct_mapping
            d = loader.get_single_data()

            fp.close()
            if d is not None:
                allinput.update(d)
        #            util.update(allinput, d)  # recursive update
        elif ext == ".json":
            raise NotImplementedError("Can not deal with json input for now")
        else:
            # process splicer file on command line, search path is not used
            splicer.get_splicer_based_on_suffix(filename, splicers)

    # Add options from command line last
    # so they replace values from YAML files.
    if args.option:
        cmdoptions = {}
        for option in args.option:
            name, value = option.split("=",1)
            if value in ["true", "True"]:
                value = True
            elif value in ["false", "False"]:
                value = False
            cmdoptions[name] = value
        if "options" in allinput:
            allinput["options"].update(cmdoptions)
        else:
            allinput["options"] = cmdoptions

    if args.language:
        allinput['language'] = args.language

    #    print(allinput)

    symtab = declast.SymbolTable()
    def_types = symtab.typemaps  #typemap.initialize()

    # Write out native types as YAML if requested
    if config.yaml_types:
        TypeOut(None, config).write_all_types(def_types, config.yaml_types)

    newlibrary = ast.create_library_from_dictionary(allinput, symtab=symtab)

    try:
        generate.generate_functions(newlibrary, config)
    except RuntimeError:
        dump_jsonfile(config, args.logdir, basename, newlibrary)
        raise

    if "splicer" in allinput:
        # read splicer files defined in input YAML file
        for suffix in sorted(allinput["splicer"].keys()):
            if suffix == "__line__":
                continue
            names = allinput["splicer"][suffix]
            # suffix = 'c', 'f', 'py', 'lua'
            subsplicer = splicers.setdefault(suffix, {})
            for name in names:
                for pth in search_path:
                    fullname = os.path.join(pth, name)
                    #                    log.write("Try splicer %s\n" % fullname)
                    if os.path.isfile(fullname):
                        break
                else:
                    fullname = None
                if not fullname:
                    raise RuntimeError("File not found: %s" % name)
                log.write("Read splicer %s\n" % name)
                splicer.get_splicers(fullname, subsplicer)

    # Add any explicit splicers in the YAML file.
    if "splicer_code" in allinput:
        splicers.update(allinput["splicer_code"])

    # Write out generated types
    TypeOut(newlibrary, config).write_class_types()

    try:
        statements.update_fc_statements_for_language(newlibrary.language)
        wrap = newlibrary.wrap

        metaattrs.process_metaattrs(newlibrary, "share")
        
        if wrap.c:
            metaattrs.process_metaattrs(newlibrary, "c")
        if wrap.fortran:
            metaattrs.process_metaattrs(newlibrary, "f")
        
        # Wrap C functions first to see which actually generate wrappers
        # based on fc_statements. Then the Fortran wrapper will call
        # the C function directly or the wrapped function.
        clibrary = wrapc.Wrapc(newlibrary, config, splicers["c"])
        if wrap.c or wrap.fortran:
            fcfmt.FillFormat(newlibrary).fmt_library()
            clibrary.wrap_library()

        if wrap.fortran:
            wrapf.Wrapf(newlibrary, config, splicers["f"]).wrap_library()

        if wrap.c or wrap.fortran:
            clibrary.write_post_fortran()

        if wrap.python:
            metaattrs.process_metaattrs(newlibrary, "py")
            wrapp.Wrapp(newlibrary, config, splicers["py"]).wrap_library()

        if wrap.lua:
            metaattrs.process_metaattrs(newlibrary, "lua")
            wrapl.Wrapl(newlibrary, config, splicers["lua"]).wrap_library()
    finally:
        # Write a debug dump even if there was an exception.
        dump_jsonfile(config, args.logdir, basename, newlibrary)
#        dump_fmt(config, args.logdir, basename, newlibrary)

    # Write list of output files.  May be useful for build systems
    if args.cfiles:
        with open(args.cfiles, "w") as fp:
            if config.cfiles:
                fp.write(" ".join(config.cfiles))
            fp.write("\n")
    if args.ffiles:
        with open(args.ffiles, "w") as fp:
            if config.ffiles:
                fp.write(" ".join(config.ffiles))
            fp.write("\n")

    if args.write_helpers:
        hfile = os.path.join(args.logdir, args.write_helpers + ".c")
        with open(hfile, "w") as fp:
            whelpers.write_c_helpers(fp)
        hfile = os.path.join(args.logdir, args.write_helpers + ".f")
        with open(hfile, "w") as fp:
            whelpers.write_f_helpers(fp)

    if args.write_statements:
        hfile = os.path.join(args.logdir, args.write_statements)

        lang = newlibrary.language
        wrapp.update_statements_for_language(lang)
        wrapl.update_statements_for_language(lang)

        with open(hfile, "w") as fp:
            fp.write("***** Fortran/C\n")
            statements.write_cf_tree(fp)
            fp.write("***** Python\n")
            wrapp.write_stmts_tree(fp)
            fp.write("***** Lua\n")
            wrapl.write_stmts_tree(fp)
            
    log.close()

    # This helps when running with a pipe, like CMake's execute_process
    # It doesn't fix the error, but it does report a better error message
    # http://www.thecodingforums.com/threads/help-with-a-piping-error.749747/
    sys.stdout.flush()
    return config


def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb) # more "modern"

# Run pdb on exception
# https://stackoverflow.com/questions/242485/starting-python-debugger-automatically-on-error
#sys.excepthook = info

if __name__ == "__main__":
    main()
