
add_mlir_dialect_library(MLIRLoxTransforms
    ShapeInferencePass.cpp

    DEPENDS
    MLIRLoxIncGen
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

