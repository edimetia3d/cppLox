add_mlir_conversion_library(LoxToMixedLox
    LoxToMixedLox.cpp
    Patterns.cpp

    LINK_LIBS PUBLIC
    MLIRLoxDialect
)

