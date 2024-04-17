add_mlir_dialect_library(MLIRLoxDialect
    LoxDialect.cpp
    LoxOps.cpp
    LoxTypes.cpp
    ConstantOpVerify.cpp
    CanonicalPatterns/TransposeRewriter.cpp
    CanonicalPatterns/ReshapeReWriter.cpp

    DEPENDS
    MLIRLoxIncGen
    MLIRLoxShapeInferInterfaceIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRCastInterfaces
    MLIRCallInterfaces
    MLIRLoxTransforms
)