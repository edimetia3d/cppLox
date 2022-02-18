cmake_minimum_required(VERSION 3.13)
project(Lox)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DSPDLOG_ACTIVE_LEVEL=1")

include(cmake/build_options.cmake)
include(cmake/win32_compat.cmake)
if (ENABLE_MLIR_JIT_BACKEND)
    cmake_policy(SET CMP0116 OLD)
    set(MLIR_DIR ${CMAKE_CURRENT_LIST_DIR}/third_party/llvm-project/install/lib/cmake/mlir)
    find_package(MLIR REQUIRED CONFIG)
    message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

    # todo: check these three dir, they ar very important for linking
    # todo: check why llvm should be built with same stdlib as lox, or linking will fail
    set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/llvm_bin)
    set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/llvm_lib)
    set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR}/mlir_bin)

    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)

    include_directories(${LLVM_INCLUDE_DIRS})
    include_directories(${MLIR_INCLUDE_DIRS})
    link_directories(${LLVM_BUILD_LIBRARY_DIR})

    include(cmake/mlir_build_options.cmake)

endif ()

add_subdirectory(third_party)
include_directories(${CLI11_DIR}/include)
include_directories(${SPDLOG_DIR}/include)

include_directories(${CMAKE_CURRENT_LIST_DIR}/includes)
include_directories(${CMAKE_CURRENT_LIST_DIR}/srcs)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/includes)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/srcs)

add_subdirectory(srcs)
add_subdirectory(bin)
