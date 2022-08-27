set(LLVM_LINK_COMPONENTS
        Core
        Support
        nativecodegen
        OrcJIT
        )

add_subdirectory(translation)

add_library(mlir_backend mlir_jit.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(mlir_backend
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
        lox_common
        ast_to_mlir
        )