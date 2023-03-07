# these option and definitions should only be used when building things related to MLIR
# we do not populate these options to global in the main CMakeLists.txt
macro(set_llvm_build_options)
    cmake_policy(SET CMP0116 OLD)
    find_package(MLIR REQUIRED CONFIG)

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

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ambiguous-reversed-operator") # ignore errors from llvm headers
    endif ()
endmacro()