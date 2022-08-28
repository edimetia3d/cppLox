add_mlir_dialect_library(MLIRLox
        LoxDialect.cpp
        LoxInterface.cpp
        ConstantOpVerify.cpp

        DEPENDS
        mlir-headers
        mlir-generic-headers

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRCastInterfaces
        MLIRCallInterfaces
        )