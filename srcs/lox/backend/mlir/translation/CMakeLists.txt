
set_llvm_build_options()

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

cc_library(translation ast_to_mlir.cc)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(lox.backend.mlir.translation PUBLIC "-Wno-ambiguous-reversed-operator") # ignore errors from llvm headers
endif ()

llvm_update_compile_flags(lox.backend.mlir.translation)
set_target_properties(lox.backend.mlir.translation PROPERTIES OUTPUT_NAME "mlir_translation")
target_link_libraries(lox.backend.mlir.translation
    PRIVATE
    ${dialect_libs}
    ${translation_libs}
    MLIRIR
    lox.common
    )

mlir_check_link_libraries(lox.backend.mlir.translation)
