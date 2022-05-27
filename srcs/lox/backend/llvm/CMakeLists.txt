set(LLVM_LINK_COMPONENTS
        Core
        Support
        nativecodegen
        OrcJIT
        )

add_subdirectory(translation)

add_library(llvm_backend llvm_jit.cpp)

target_link_libraries(llvm_backend
        PRIVATE
        lox_common
        ast_to_llvm
        )