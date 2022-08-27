
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_library(lox_transforms
        TransposeRewriter.cpp
        ReshapeReWriter.cpp
        ShapeInferencePass.cpp
        )
llvm_update_compile_flags(lox_transforms)
target_link_libraries(lox_transforms
        PUBLIC
        ${dialect_libs}
        MLIRIR
        MLIRPass
        MLIRSupport
        MLIRAnalysis
        MLIRSideEffectInterfaces
        MLIRTransforms
        )

mlir_check_link_libraries(lox_transforms)

