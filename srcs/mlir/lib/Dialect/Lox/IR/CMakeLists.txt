add_mlir_dialect_library(MLIRLoxDialect
    LoxDialect.cpp
    LoxOps.cpp
    LoxTypes.cpp
    InlinerInterface.cpp
    ConstantOpVerify.cpp

    DEPENDS
    MLIRLoxIncGen
    MLIRLoxShapeInferInterfaceIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRCastInterfaces
    MLIRCallInterfaces
    MLIRLoxTransforms
    )