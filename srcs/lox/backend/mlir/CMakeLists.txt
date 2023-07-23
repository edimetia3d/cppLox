include(third_party_llvm)
setup_llvm()

set(LLVM_LINK_COMPONENTS
    Core
    Support
    nativecodegen
    OrcJIT
)

# fixme: update source code so that -Wl,-z,defs could be applied
message(DEBUG "Linker flags: ${CMAKE_SHARED_LINKER_FLAGS} changed to \"-Wl,-z,nodelete\"")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,nodelete")

add_subdirectory(translation)

cc_library(mlir SHARED mlir_jit.cpp)


get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
message(DEBUG "dialect_libs: ${dialect_libs}")
message(DEBUG "conversion_libs: ${conversion_libs}")
target_link_libraries(lox.backend.mlir
    PRIVATE
    MLIRLoxAll
    ${dialect_libs}
    ${conversion_libs}
    ${translation_libs}
    ${extension_libs}
    MLIRExecutionEngine
    MLIRIR
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRTargetLLVMIRExport
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRAnalysis
    MLIRSideEffectInterfaces
    MLIRTransforms
    lox.common
    lox.backend.mlir.translation
)
llvm_update_compile_flags(lox.backend.mlir)
mlir_check_all_link_libraries(lox.backend.mlir)