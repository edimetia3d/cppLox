
add_subdirectory(ast)
add_subdirectory(passes)
add_library(frontend scanner.cc token.cc parser.cc)
target_link_libraries(frontend PUBLIC ast passes)
