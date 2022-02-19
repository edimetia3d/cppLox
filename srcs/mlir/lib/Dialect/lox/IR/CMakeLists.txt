add_mlir_dialect_library(MLIRLox
        Dialect.cpp

        DEPENDS
        MLIRloxIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )