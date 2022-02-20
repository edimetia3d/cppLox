
set_mlir_build_options()

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_library(ast_to_mlir SHARED
        ast_to_mlir.cc
        ast_to_mlir.h
        )

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(ast_to_mlir PUBLIC "-Wno-ambiguous-reversed-operator") # ignore errors from llvm headers
endif ()

llvm_update_compile_flags(ast_to_mlir)

target_link_libraries(ast_to_mlir
        PUBLIC
        ${dialect_libs}
        ${translation_libs}
        MLIRIR
        lox_common
        )

mlir_check_link_libraries(ast_to_mlir)
