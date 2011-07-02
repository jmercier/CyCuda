#
# Cython
#

# This finds the "cython" executable in your PATH, and then in some standard
# paths:
FIND_FILE(CYTHON_BIN cython /usr/bin /usr/local/bin)
SET(CYTHON_FLAGS)

SET(Cython_FOUND CYTHON_BIN)
IF (CYTHON_BIN)
    # Try to run Cython, to make sure it works:
    #execute_process(
    #    COMMAND ${CYTHON_BIN} ${CYTHON_FLAGS} ${CMAKE_MODULE_PATH}/cython_test.pyx
    #    RESULT_VARIABLE CYTHON_RESULT
    #    OUTPUT_QUIET
    #    ERROR_QUIET
    #    )
    #if (CYTHON_RESULT EQUAL 0)
    #    # Only if cython exits with the return code 0, we know that all is ok:
    #    SET(Cython_FOUND TRUE)
    #    SET(Cython_Compilation_Failed FALSE)
    #else (CYTHON_RESULT EQUAL 0)
    #    SET(Cython_Compilation_Failed TRUE)
    #endif (CYTHON_RESULT EQUAL 0)
ENDIF (CYTHON_BIN)


IF (Cython_FOUND)
	IF (NOT Cython_FIND_QUIETLY)
		MESSAGE(STATUS "Found Cython: ${CYTHON_BIN}")
	ENDIF (NOT Cython_FIND_QUIETLY)
ELSE (Cython_FOUND)
	IF (Cython_FIND_REQUIRED)
        if(Cython_Compilation_Failed)
            MESSAGE(FATAL_ERROR "Your Cython version is too old. Please upgrade Cython.")
        else(Cython_Compilation_Failed)
            MESSAGE(FATAL_ERROR "Could not find Cython. Please install Cython.")
        endif(Cython_Compilation_Failed)
	ENDIF (Cython_FIND_REQUIRED)
ENDIF (Cython_FOUND)


# This allows to link Cython files
# Examples:
# 1) to compile assembly.pyx to assembly.so:
#   CYTHON_ADD_MODULE(assembly)
# 2) to compile assembly.pyx and something.cpp to assembly.so:
#   CYTHON_ADD_MODULE(assembly something.cpp)

if(NOT CYTHON_INCLUDE_DIR)
    set(CYTHON_INCLUDE_DIR .)
endif(NOT CYTHON_INCLUDE_DIR)

# Cythonizes the .pyx files into .cpp file (but doesn't compile it)
function(CYTHON_ADD_MODULE_PYX name)
    add_custom_command(
        OUTPUT ${name}.c
        COMMAND ${CYTHON_BIN}
        ARGS ${CYTHON_FLAGS} -I ${CYTHON_INCLUDE_DIR} -o ${name}.c ${name}.pyx
        DEPENDS ${name}.pyx
        COMMENT "Cythonizing ${name}.pyx")
endfunction(CYTHON_ADD_MODULE_PYX)

# Cythonizes and compiles a .pyx file

function(CYTHON_ADD_MODULE name)
    CYTHON_ADD_MODULE_PYX(${name})
    # We need Python for this:
    FIND_PACKAGE(PythonInterp REQUIRED)
    FIND_PACKAGE(PythonLibs REQUIRED)
    PYTHON_ADD_MODULE(${name} ${ARGN})
endfunction(CYTHON_ADD_MODULE)

