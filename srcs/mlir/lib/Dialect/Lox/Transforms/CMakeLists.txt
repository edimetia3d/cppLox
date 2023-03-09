
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_library(MLIRLoxTransforms
    TransposeRewriter.cpp
    ReshapeReWriter.cpp
    ShapeInferencePass.cpp
    )
llvm_update_compile_flags(MLIRLoxTransforms)
target_link_libraries(MLIRLoxTransforms
    PUBLIC
    ${dialect_libs}
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRAnalysis
    MLIRSideEffectInterfaces
    MLIRTransforms
    MLIRLox
    )

mlir_check_link_libraries(MLIRLoxTransforms)

