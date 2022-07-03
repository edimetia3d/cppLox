# these option and definitions should only be used when building things related to MLIR
# we do not populate these options to global in the main CMakeLists.txt
macro(set_llvm_build_options)
    include(HandleLLVMOptions)
    add_definitions(${LLVM_DEFINITIONS})
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ambiguous-reversed-operator") # ignore errors from llvm headers
    endif ()
endmacro()