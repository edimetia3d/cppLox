add_mlir_conversion_library(MixedLoxToLLVM
    MixedLoxToLLVM.cpp
    Patterns.cpp

    DEPENDS
    MLIRMixedLoxToLLVMIncGen

    LINK_LIBS PUBLIC
    MLIRLoxDialect
    MLIRAffineToStandard
    MLIRSCFToControlFlow

)