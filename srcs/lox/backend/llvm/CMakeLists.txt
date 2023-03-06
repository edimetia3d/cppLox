set(LLVM_LINK_COMPONENTS
        Core
        Support
        nativecodegen
        OrcJIT
        native
        )

add_subdirectory(translation)
add_subdirectory(builtins)


cc_library(llvm SHARED llvm_jit.cpp)

target_link_libraries(lox.backend.llvm
        PRIVATE
        lox.common
        lox.backend.llvm.translation
        lox.backend.llvm.builtins
        )