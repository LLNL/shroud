

# gmake

```
mkdir gtest
cd gtest
make -f ../src/GNUmakefile shroud.exe=../../build/temp.linux-x86_64-2.7/venv/bin/shroud
```

# CMake

```
mkdir ctest
cd ctest
cmake ../src SHROUD_EXE=../../build/temp.linux-x86_64-2.7/venv/bin/shroud
```