cmake_minimum_required(VERSION 3.18)
project(Lox)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DSPDLOG_ACTIVE_LEVEL=1")

include(CTest)
include(build_options)
include(win32_compat)
include(build_fn)
include(prepare_third_party)

if (ENABLE_JIT_BACKEND)
    set(MLIR_DIR ${CMAKE_CURRENT_LIST_DIR}/third_party/llvm-project/install/lib/cmake/mlir CACHE PATH "MLIR install prefix")
endif ()


include_directories(${CMAKE_CURRENT_LIST_DIR}/includes)
include_directories(${CMAKE_CURRENT_LIST_DIR}/srcs)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/includes)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/srcs)

add_subdirectory(srcs)
add_subdirectory(bin)
