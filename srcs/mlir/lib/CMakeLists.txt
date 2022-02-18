
add_library(lox_dialect Dialect/lox/IR/Dialect.cpp)

add_mlir_dialect_library(MLIRLox
        Dialect/lox/IR/Dialect.cpp

        DEPENDS
        MLIRloxIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )