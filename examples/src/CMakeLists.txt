cmake_minimum_required(VERSION 3.1.0)
enable_language(C Fortran)

if(EXISTS ${SHROUD_EXECUTABLE})

    # Shroud Setup
    execute_process(COMMAND ${SHROUD_EXECUTABLE}
                    --cmake ${CMAKE_CURRENT_BINARY_DIR}/SetupShroud.cmake
                    ERROR_VARIABLE SHROUD_cmake_error
                    OUTPUT_STRIP_TRAILING_WHITESPACE )
    if(${SHROUD_cmake_error})
       message(FATAL_ERROR "Error from Shroud: ${SHROUD_cmake_error}")
    endif()
    include(${CMAKE_CURRENT_BINARY_DIR}/SetupShroud.cmake)

    #NOTE: There is an issue where giving SHROUD_YAML_FILE the absolute path to an out of directory yaml file will cause 
    #      a compile time error. This is probably an issue with how add_shroud() parses and handles the YAML_INPUT_FILE argument.
    if (EXISTS ${SHROUD_YAML_FILE})
        add_shroud(YAML_INPUT_FILE ${SHROUD_YAML_FILE}
                   C_FORTRAN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                   PYTHON_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
    )
    else ()
        message(FATAL_ERROR "Shroud requires a valid yaml input file. This may be set in CMake under SHROUD_YAML_FILE.")
    endif()

    # Library source files being wrapped
    file(GLOB LIB_SRC foo.cpp bar.cpp)
    set(LIB_INC ${CMAKE_CURRENT_SOURCE_DIR})

    # Fortran Wrappers
    set(BUILD_FORTRAN_WRAPPERS on CACHE BOOL "Build Fortran Library Wrapper")
    if (BUILD_FORTRAN_WRAPPERS)
        set_source_files_properties(wrap${_basename}.cpp PROPERTIES GENERATED TRUE)
        set_source_files_properties(wrapf${_basename}.f PROPERTIES GENERATED TRUE)
        set(CMAKE_Fortran_FORMAT FREE)

        add_library(${_basename}_f STATIC wrapf${_basename}.f wrap${_basename}.cpp ${LIB_SRC})

        target_include_directories(${_basename}_f PRIVATE ${LIB_INC})
        set_target_properties(${_basename}_f PROPERTIES LINKER_LANGUAGE Fortran)
        set_target_properties(${_basename}_f PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
        set_target_properties(${_basename}_f PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/include)

        add_dependencies(${_basename}_f generate_${_basename})
    endif()
    
    # Python Wrappers
    set(BUILD_PYTHON_WRAPPERS on CACHE BOOL "Build Python Library Wrapper")
    if (BUILD_PYTHON_WRAPPERS)
        find_package(PythonInterp 3.7 REQUIRED) # Change to 3.X if using strictly python 3
        if(PYTHONINTERP_FOUND)
            exec_program(${PYTHON_EXECUTABLE}
                        ARGS "-c \"from sysconfig import get_paths; print(get_paths()['include'])\""
                        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
            )
            message(STATUS "Found Python headers: ${PYTHON_INCLUDE_DIR}")

            exec_program(${PYTHON_EXECUTABLE}
                         ARGS "-c \"import numpy; print(numpy.get_include())\""
                         OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
                         RETURN_VALUE NUMPY_NOT_FOUND
            )
            if(NUMPY_NOT_FOUND)
                message(FATAL_ERROR "NumPy not found")
            else()
                message(STATUS "Found NumPy headers: ${NUMPY_INCLUDE_DIR}")

                set_source_files_properties(py${_basename}util.cpp PROPERTIES GENERATED TRUE)
                set_source_files_properties(py${_basename}module.cpp PROPERTIES GENERATED TRUE)

                add_library(${_basename}_py SHARED py${_basename}util.cpp py${_basename}module.cpp ${LIB_SRC})

                target_include_directories(${_basename}_py PRIVATE ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIR} ${LIB_INC})
                set_target_properties(${_basename}_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
                set_target_properties(${_basename}_py PROPERTIES OUTPUT_NAME "${_basename}")
                set_target_properties(${_basename}_py PROPERTIES PREFIX "")

                add_dependencies(${_basename}_py generate_${_basename})

                file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/lib/__init__.py CONTENT "")
            endif()
        else()
            message(FATAL_ERROR "Python not found")
        endif()
    endif()

else()
    message(FATAL_ERROR "Requires path to Shroud. This can be set in CMake under SHROUD_EXECUTABLE.")
endif()
