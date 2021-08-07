
add_library(environment
        environment.cpp
        )

add_library(expr_stmt
        expr.cpp
        stmt.cpp
        )

add_library(ast_printer ast_printer.cpp)
target_link_libraries(ast_printer PRIVATE lox_object)
target_link_libraries(ast_printer PRIVATE expr_stmt)

add_library(evaluator eval_visitor.cpp)
target_link_libraries(evaluator PRIVATE lox_object)
target_link_libraries(evaluator PRIVATE expr_stmt)
target_link_libraries(evaluator PRIVATE environment)

add_library(ast INTERFACE)
target_link_libraries(ast INTERFACE evaluator ast_printer expr_stmt)


find_package(Python3 COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
    target_compile_definitions(expr_stmt PUBLIC DYNAMIC_GEN_DECL)

    add_custom_target(generated_expr_decl
            COMMAND ${Python3_EXECUTABLE} nodes_code_writer.py ${CMAKE_CURRENT_LIST_DIR}/expr_stmt_def.tpl ${CMAKE_CURRENT_BINARY_DIR}/expr_decl_dynamic.h.inc Expr
            COMMAND ./update_local_cache.sh ${CMAKE_CURRENT_BINARY_DIR}/expr_decl_dynamic.h.inc ${CMAKE_CURRENT_LIST_DIR}/expr_decl.h.inc
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/)
    add_dependencies(ast generated_expr_decl)

    add_custom_target(generated_stmt_decl
            COMMAND ${Python3_EXECUTABLE} nodes_code_writer.py ${CMAKE_CURRENT_LIST_DIR}/expr_stmt_def.tpl ${CMAKE_CURRENT_BINARY_DIR}/stmt_decl_dynamic.h.inc Stmt
            COMMAND ./update_local_cache.sh ${CMAKE_CURRENT_BINARY_DIR}/stmt_decl_dynamic.h.inc ${CMAKE_CURRENT_LIST_DIR}/stmt_decl.h.inc
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/)
    add_dependencies(ast generated_stmt_decl)

else ()
    message(WARNING "Using cached expr_decl, it might be out of sync!")
endif ()
