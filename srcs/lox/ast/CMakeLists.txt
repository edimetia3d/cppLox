
add_library(ast
        expr.h
        stmt.h
        )

target_link_libraries(ast PUBLIC lox_object)
find_package(Python3 COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
    target_compile_definitions(ast PUBLIC DYNAMIC_GEN_DECL)

    add_custom_target(generated_expr_decl
            COMMAND ${Python3_EXECUTABLE} nodes_code_writer.py ${CMAKE_CURRENT_LIST_DIR}/expr_stmt_def.tpl ${CMAKE_CURRENT_BINARY_DIR}/expr_decl_dynamic.h.inc.tmp Expr
            COMMAND ./update_local_cache.sh ${CMAKE_CURRENT_BINARY_DIR}/expr_decl_dynamic.h.inc.tmp ${CMAKE_CURRENT_LIST_DIR}/expr_decl.h.inc
            COMMAND ./update_local_cache.sh ${CMAKE_CURRENT_BINARY_DIR}/expr_decl_dynamic.h.inc.tmp ${CMAKE_CURRENT_BINARY_DIR}/expr_decl_dynamic.h.inc
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/)
    add_dependencies(ast generated_expr_decl)

    add_custom_target(generated_stmt_decl
            COMMAND ${Python3_EXECUTABLE} nodes_code_writer.py ${CMAKE_CURRENT_LIST_DIR}/expr_stmt_def.tpl ${CMAKE_CURRENT_BINARY_DIR}/stmt_decl_dynamic.h.inc.tmp Stmt
            COMMAND ./update_local_cache.sh ${CMAKE_CURRENT_BINARY_DIR}/stmt_decl_dynamic.h.inc.tmp ${CMAKE_CURRENT_LIST_DIR}/stmt_decl.h.inc
            COMMAND ./update_local_cache.sh ${CMAKE_CURRENT_BINARY_DIR}/stmt_decl_dynamic.h.inc.tmp ${CMAKE_CURRENT_BINARY_DIR}/stmt_decl_dynamic.h.inc
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/)
    add_dependencies(ast generated_stmt_decl)

else ()
    message(WARNING "Using cached expr_decl, it might be out of sync!")
endif ()
