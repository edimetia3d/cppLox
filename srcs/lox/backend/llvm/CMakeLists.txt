set(LLVM_LINK_COMPONENTS
        Core
        Support
        nativecodegen
        OrcJIT
        native
        )

add_subdirectory(translation)
add_subdirectory(builtins)


add_llvm_library(llvm_backend llvm_jit.cpp)

target_link_libraries(llvm_backend
        PRIVATE
        lox_common
        ast_to_llvm
        lox::backend::jit::builtins
        )