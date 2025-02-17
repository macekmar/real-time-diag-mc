# Version number of the application
set (CTINT_KELDYSH_VERSION "1.4")

# Append triqs installed files to the cmake load path
list(APPEND CMAKE_MODULE_PATH ${TRIQS_PATH}/share/triqs/cmake)

# start configuration 
cmake_minimum_required(VERSION 2.8)
project(ctint_keldysh CXX)
enable_testing()

# We use shared libraries
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Load TRIQS, including all predefined variables from TRIQS installation
find_package(TRIQS REQUIRED)
add_definitions(${TRIQS_CXX_DEFINITIONS})

# Check that versions are compatible
if(NOT CTINT_KELDYSH_VERSION EQUAL TRIQS_VERSION)
  message(FATAL_ERROR "The application version is not compatible with the TRIQS library (TRIQS library version: ${TRIQS_VERSION} while this application version: ${CTINT_KELDYSH_VERSION})")
endif()

# We want to be installed in the TRIQS tree
set(CMAKE_INSTALL_PREFIX ${TRIQS_PATH})

# Get hash
triqs_get_git_hash(${CMAKE_SOURCE_DIR} "CTHYB_SEGMENT")
if(${GIT_RESULT} EQUAL 0)
  message(STATUS "Hash: ${CTHYB_SEGMENT_GIT_HASH}")
endif(${GIT_RESULT} EQUAL 0)

include_directories(c++)

# add cmake modules path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

# Compile C++ code
add_subdirectory(c++)

# Python interface
if (${TRIQS_WITH_PYTHON_SUPPORT})
 add_subdirectory(python)
# option(BUILD_DOC "Build documentation" OFF)
# if(${BUILD_DOC})
#   if(NOT TRIQS_WITH_DOCUMENTATION)
#     message("Error: TRIQS library has not been compiled with its documentation")
#   endif()
#   add_subdirectory(doc)
# endif(${BUILD_DOC})
endif()

# demo example
add_subdirectory(demo)

# tests
add_subdirectory(test)

