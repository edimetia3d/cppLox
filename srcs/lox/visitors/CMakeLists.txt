
add_subdirectory(ast_printer)
add_subdirectory(evaluator)
add_subdirectory(resolver_pass)

add_library(visitors INTERFACE)
target_link_libraries(visitors INTERFACE ast_printer evaluator resolver)
