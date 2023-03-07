include(mlir_build_options)
set_llvm_build_options()

set(LLVM_LINK_COMPONENTS
    Core
    Support
    nativecodegen
    OrcJIT
    )

add_subdirectory(translation)

cc_library(mlir SHARED mlir_jit.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(lox.backend.mlir
    PRIVATE
    ${dialect_libs}
    ${translation_libs}
    ${conversion_libs}
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRTargetLLVMIRExport
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRAnalysis
    MLIRSideEffectInterfaces
    MLIRTransforms
    MLIRLox
    lox.common
    lox.backend.mlir.translation
    )