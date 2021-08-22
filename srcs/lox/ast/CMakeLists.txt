
add_library(ast
        ast.cc
        ast.h
        )

target_link_libraries(ast PUBLIC lox_object)
find_package(Python3 COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
    target_compile_definitions(ast PUBLIC DYNAMIC_GEN_DECL)

    add_custom_target(generated_ast_decl
            COMMAND ${Python3_EXECUTABLE} nodes_code_writer.py ${CMAKE_CURRENT_LIST_DIR}/ast_node_def.tpl ${CMAKE_CURRENT_BINARY_DIR}/ast_decl_dynamic.h.inc.tmp
            COMMAND ./update_local_cache.sh ${CMAKE_CURRENT_BINARY_DIR}/ast_decl_dynamic.h.inc.tmp ${CMAKE_CURRENT_LIST_DIR}/ast_decl.h.inc
            COMMAND ./update_local_cache.sh ${CMAKE_CURRENT_BINARY_DIR}/ast_decl_dynamic.h.inc.tmp ${CMAKE_CURRENT_BINARY_DIR}/ast_decl_dynamic.h.inc
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/)
    add_dependencies(ast generated_ast_decl)

else ()
    message(WARNING "Using cached ast_decl, it might be out of sync!")
endif ()
