
set_llvm_build_options()

set(LLVM_LINK_COMPONENTS
        Support
        Core
        )

add_llvm_library(ast_to_llvm SHARED
        ast_to_llvm.cpp
        ast_to_llvm.h
        )

target_link_libraries(ast_to_llvm
        PUBLIC
        lox_common
        )

llvm_update_compile_flags(ast_to_llvm)
