
add_library(ast
        ast_node.cc
        ast.h
        )

find_package(Python3 COMPONENTS Interpreter)
find_program(BASH_PROGRAM bash)
if (Python3_Interpreter_FOUND AND BASH_PROGRAM)

    add_custom_target(generated_ast_nodes_decl
            COMMAND bash gen_tool/update_local_cache.sh ${CMAKE_CURRENT_LIST_DIR}/ast_node_def.yaml ast_nodes_decl.h.inc ${CMAKE_CURRENT_BINARY_DIR}
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
    add_dependencies(ast generated_ast_nodes_decl)

else ()
    message(WARNING "Using cached ast_nodes_decl, it might be out of sync!")
endif ()

add_subdirectory(ast_printer)

target_link_libraries(ast PUBLIC ast_printer)
