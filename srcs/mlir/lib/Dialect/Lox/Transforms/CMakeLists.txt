
add_mlir_dialect_library(MLIRLoxTransforms
    TransposeRewriter.cpp
    ReshapeReWriter.cpp
    ShapeInferencePass.cpp

    DEPENDS
    MLIRLoxIncGen
    MLIRLoxReshapeRewriterIncGen
    MLIRLoxShapeInferInterfaceIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRAnalysis
    MLIRSideEffectInterfaces
    MLIRTransforms
    )

mlir_check_link_libraries(MLIRLoxTransforms)

