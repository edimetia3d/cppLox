include(third_party_llvm)
setup_llvm()

# fixme: update source code so that -Wl,-z,defs could be applied
message(DEBUG "Linker flags: ${CMAKE_SHARED_LINKER_FLAGS} changed to \"-Wl,-z,nodelete\"")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,nodelete")


add_subdirectory(translation)
add_subdirectory(builtins)

cc_library(llvm SHARED llvm_jit.cpp)

llvm_map_components_to_libnames(llvm_libs passes support core nativecodegen OrcJIT native)

target_link_libraries(lox.backend.llvm
    PRIVATE
    ${llvm_libs}
    lox.common
    lox.backend.llvm.translation
    lox.backend.llvm.builtins
    )
llvm_update_compile_flags(lox.backend.llvm)