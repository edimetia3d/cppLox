
add_library(ast_printer ast_printer.cpp)
target_link_libraries(ast_printer PUBLIC ast)
target_link_libraries(ast_printer PUBLIC lox_object)
