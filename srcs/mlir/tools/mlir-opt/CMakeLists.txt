
set(LLVM_OPTIONAL_SOURCES
    null.cpp
)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
    Core
    Support
    AsmParser
)

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRAffineAnalysis
    MLIRAnalysis
    MLIRDialect
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRSupport
    MLIRIR
)

add_mlir_tool(lox-mlir-opt
    mlir-opt.cc
    DEPENDS
    ${LIBS}
)

target_link_libraries(lox-mlir-opt PRIVATE ${LIBS})
llvm_update_compile_flags(lox-mlir-opt)
mlir_check_all_link_libraries(lox-mlir-opt)
