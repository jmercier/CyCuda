cmake_minimum_required (VERSION 2.6)

project (CyCuda)

set (CyCuda_VERSION_MAJOR 0)
set (CyCuda_VERSION_MINOR 5)

set (CMAKE_MODULE_PATH ${CyCuda_SOURCE_DIR}/cmake)

set (CMAKE_C_FLAGS "-O3 -DNDEBUG")

FIND_PACKAGE(Mako REQUIRED)
FIND_PACKAGE(Cython REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)

FILE(GLOB CyCuda_MODULE "${CyCuda_SOURCE_DIR}/cycuda/*.cpp")
SET (CMAKE_C_FLAGS                "-Wall -Wno-unused-value -Wno-unused-function")

set (CyCuda_TEMPLATE_DIR ${CyCuda_SOURCE_DIR}/templates)
set (MAKO_TEMPLATE_COMMAND ${CyCuda_SOURCE_DIR}/scripts/gen_template.py ${INPUT} ${OUTPUT})
set (CyCuda_LICENSE ${CyCuda_SOURCE_DIR}/LICENSE)
#set (Cython_CFLAGS "-fno-strict-aliasing -fwrapv")
SET (MAKO_TEMPLATE_DIR ${CyCuda_TEMPLATE_DIR})
SET (MAKO_EXTRA_ARGS --license ${CyCuda_LICENSE})

MAKO_ADD_MODULE("fft.pyx")
MAKO_ADD_MODULE("core.pyx")
MAKO_ADD_MODULE("garray.pyx")
MAKO_ADD_MODULE("random.pyx")

set (CYTHON_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src/cycuda -I${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES( ${CUDA_INCLUDE_DIRS} )

#CYTHON_ADD_MODULE(cycuda/garray garray garray.c)
CYTHON_ADD_MODULE(cycuda/core core  core.c)
CYTHON_ADD_MODULE(cycuda/fft fft fft.c)
CYTHON_ADD_MODULE(cycuda/random random random.c)

#add_custom_target(cycuda COMMAND cp -r ${PROJECT_SOURCE_DIR}/cycuda cycuda DEPENDS ${CyCuda_MODULE})

TARGET_LINK_LIBRARIES(cycuda/core "cuda")
SET_TARGET_PROPERTIES(cycuda/core PROPERTIES INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")
ADD_DEPENDENCIES(cycuda/core cycuda)

#TARGET_LINK_LIBRARIES(cycuda/garray "cuda")
#SET_TARGET_PROPERTIES(cycuda/garray PROPERTIES LINK_FLAGS "-L/usr/lib/nvidia-current")
#ADD_DEPENDENCIES(cycuda/garray cycuda)

MESSAGE(${CUDA_INCLUDE_DIRS})

FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cycuda DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

TARGET_LINK_LIBRARIES(cycuda/fft "cuda" ${CUDA_CUFFT_LIBRARIES})
SET_TARGET_PROPERTIES(cycuda/fft PROPERTIES LINK_FLAGS "-L/usr/lib/nvidia-current")
ADD_DEPENDENCIES(cycuda/fft cycuda)

TARGET_LINK_LIBRARIES(cycuda/random "cuda")
SET_TARGET_PROPERTIES(cycuda/random PROPERTIES LINK_FLAGS "-L/usr/lib/nvidia-current")
ADD_DEPENDENCIES(cycuda/random cycuda)





