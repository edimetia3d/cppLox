add_mlir_dialect_library(MLIRLox
        Dialect.cpp

        DEPENDS
        MLIRloxIncGen
        MLIRShapeInferenceInterfaceIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRCastInterfaces
        MLIRCallInterfaces
        )