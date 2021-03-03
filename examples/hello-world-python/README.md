# Hello World Python

The is a complete "hello world" Python example to show you how to use Shroud.
A development environment (container) is provided if needed, otherwise you
should look at the [Dockerfile](Dockerfile) and ensure that you have basic d
dependencies installed.

## Getting Started

### 0. High Level Plan

Since a lot of the work has been done (and is just shown in the repository) let's
review how you would start this from scratch. When using shroud, we want to do the following

1. Write your C++ script, let's call it helloworld.cpp.
2. Write your header file for the script, helloworld.hpp.
3. Create a shroud input file, helloworld.yaml (note that if you have an older version of shroud the extension .yml will not work as expected)
4. Run shroud to generate your Python files.
5. Build your Python files
6. Test the Python library

These steps wil be explained in detail below.


### 1. Development Environment

Let's first build our development container.

```bash
$ docker build -t helloworld .
```

We will want to bind the code directory (to write files and work interactively):

```bash
$ docker run -it --rm -v $PWD/:/code/ helloworld bash
```

If you don't want to use a container this is fine, just make sure you have basic
dependencies on your host for compiling (build-essential) and a g++ compiler,
along with shroud. At the time of this writing, installing from the master branch
of the repository at [https://github.com/llnl/shroud](https://github.com/llnl/shroud)
is your best bet, as the release on pypi has a bug with strings, and the extensions
of yaml files.

### 2. Write your C++ Scripts

You can do this as your normally would without shroud. Shroud can handle namespaces
and classes, and the example scripts show that well:


 - [helloworld.cpp](helloworld.cpp)
 - [helloworld.hpp](helloworld.hpp)
  
The reason they are in the same directory with the same name is that when I tried
doing this differently, even when I added the cpp file to sources (which we will
discuss later) it told me there was a missing symbol after build. This is
the easiest way to ensure that doesn't happen.

### 3. Write your shroud yaml file

Shroud works by having you provide function and type signatures in a yaml file.
You can also specify build options like "Please build Python bindings for this."
Here is an example [helloworld.yaml](helloworld.yaml) that will generate for 
our small "Person" class with two functions:

```yaml
library: helloworld

# This must be relative to python directory targeted by shroud
cxx_header: ../helloworld.hpp

options:
  wrap_fortran: False
  wrap_c: False
  wrap_python: True
  debug: True

declarations:
- decl: namespace helloworld
  declarations:
  - decl: class Person
    declarations:
    - decl: int SayHello()
    - decl: int NamedHello(std::string path)
```
This is a fairly small example, and you can look at the [shroud documentation](https://shroud.readthedocs.io/en/develop/tutorial.html)
for further types. If you see something missing, please [open an issue](https://github.com/llnl/shroud) as it is
still a work in progress.

### 3. Compiling

Ok, great! We have our source flies, and the shroud yaml. First, let me show you how to compile the files,
separately from shroud. The commands are defined in the [Makefile](Makefile) and you will generate a binary,
object file, and shared object file in the [bin](bin) directory (these are not added to the repository here
as they are host dependent).

```bash
$ make
g++ -o bin/helloworld -Iinclude helloworld.cpp -g -Wall 
g++ -c -o bin/helloworld.o -fPIC -Iinclude helloworld.cpp -g -Wall 
g++ -shared -o bin/helloworld.so bin/helloworld.o
```
```bash

You could then try running the binary file:

```bash
root@b45e195a34dd:/code# bin/helloworld
Hello Dinosaur!
```

The next step is to generate the setup.py (and other python files) with shroud.
This is also provided as a command in the makefile:

```bash
# output folder must exist for Python files
$ mkdir -p helloworld
$ make shroud
```

which looks like this:

```bash
cp setup.py _setup.py | true
shroud helloworld.yaml --outdir-python helloworld
cp _setup.py setup.py | true
```

The above is a simple, non elegant way to ensure that we are not re-writing our setup.py
file, given that it exists. You'll see a bunch of Python supporting files generated in
the [helloworld](helloworld) folder. We do this to keep the namespaces separate and the repository
better organized.

```bash
 make shroud
cp setup.py _setup.py | true
shroud helloworld.yaml --outdir-python helloworld
Wrote helloworld_types.yaml
Wrote pyhelloworld_Persontype.cpp
Wrote pyhelloworld_helloworldmodule.cpp
Wrote pyhelloworldmodule.cpp
Wrote pyhelloworldutil.cpp
Wrote pyhelloworldmodule.hpp
Wrote setup.py
cp _setup.py setup.py | true
```

We are close! At this point we actually want to edit the setup.py script to include
our .cpp file as a source (and this is why we don't want to overwrite it). If we were
to compile (shown next) and *not* include the source file, we'd reliably get this error:

```bash
n [1]: import helloworld
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
<ipython-input-1-39f3e3c18221> in <module>
----> 1 import helloworld

ImportError: /usr/local/lib/python3.8/dist-packages/helloworld-0.0.0-py3.8-linux-x86_64.egg/helloworld.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZN10helloworld6Person10NamedHelloENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```
So let's edit the [setup.py](setup.py). If we add other includes or libraries we would
further edit the commented out sections.

```python
# setup.py
# This file is generated by Shroud 0.12.2. Do not edit.
from setuptools import setup, Extension

module = Extension(
    'helloworld',
    sources=[
         'helloworld/pyhelloworld_Persontype.cpp',
         'helloworld/pyhelloworld_helloworldmodule.cpp',
         'helloworld/pyhelloworldmodule.cpp',
         'helloworld/pyhelloworldutil.cpp',
         'helloworld.cpp'  # <--- we added this line
    ],
    language='c++',
    include_dirs = None,
#    libraries = ['tcl83'],
#    library_dirs = ['/usr/local/lib'],      
#    extra_compile_args = [ '-O0', '-g' ],
#    extra_link_args =
)

setup(
    name='helloworld',
    ext_modules = [module],
)
```

Building looks like this:

```bash
$ python3 setup.py install
$ python3 setup.py build # this also works without installing
```

If you need to build again, you can remove the old build and dist folders:

```bash
$ rm -rf build dist
```

**Importantly** shroud doesn't seem
to handle having *.cpp files separate from the *.hpp - if you separate them the
library will compile and report a missing symbol (see [this issue](https://github.com/LLNL/shroud/issues/223)). This is why the
structure here is flattened. It's also important that we install from GitHub master,
because the current release on pypi at the time of writing (llnl-shroud) has an error with strings.

## Python

We're done! Let's run ipython (installed in the container) and test our Python bindings!

```bash
In [1]: import helloworld

In [2]: person = helloworld.helloworld.Person()

In [3]: person.SayHello()
Hello!
Out[3]: 0

In [4]: person.NamedHello('Dinosaur')
Hello Dinosaur!
Out[4]: 0
```

Notice that since these functions return an int (0) on success that's what we get back.
Of course you would customize this further for a more real world use case.
Great job! That's all you have to do! Shroud works really nicely as long as you
don't run into any of these hiccups. We hope that you enjoy using it!
