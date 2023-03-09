add_mlir_dialect_library(MLIRLox
        LoxDialect.cpp
        LoxOps.cpp
        LoxTypes.cpp
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
target_link_libraries(MLIRLox PUBLIC MLIRLoxTransforms)