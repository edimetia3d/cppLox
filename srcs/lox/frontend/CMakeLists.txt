
add_subdirectory(ast)
add_library(frontend scanner.cc token.cc)
target_link_libraries(frontend PUBLIC ast)
