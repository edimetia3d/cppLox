
add_subdirectory(Dialect)
add_subdirectory(Conversion)

target_link_libraries(MLIRLox PUBLIC mixedlox_to_llvm lox_to_mixedlox)