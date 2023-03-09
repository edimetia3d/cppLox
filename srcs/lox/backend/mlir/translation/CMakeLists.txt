
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

cc_library(translation ast_to_mlir.cc)
set_target_properties(lox.backend.mlir.translation PROPERTIES OUTPUT_NAME "mlir_translation")
llvm_update_compile_flags(lox.backend.mlir.translation)

target_link_libraries(lox.backend.mlir.translation
    PUBLIC
    ${dialect_libs}
    ${translation_libs}
    ${conversion_libs}
    MLIRIR
    MLIRLoxAll
    lox.common
    )

mlir_check_all_link_libraries(lox.backend.mlir.translation)
